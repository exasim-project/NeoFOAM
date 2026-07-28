# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""PIMPLE pressure-velocity coupling (minimal port).

Adapted from ``feat/python_solvers`` to the SolverSpec/ModelSpec API in
``stack/python_arch``. Includes the boussinesq momentum/continuity
variants so the buoyancy plugin can swap them in via
``pimple.use_boussinesq``.

Each operation declares the ``system/fvSchemes`` entries it discretises
and the ``system/fvSolution`` solvers it needs via
``@PimpleFvSchemes.add(...)`` / ``@PimpleFvSolution.add(...)``. The
per-spec slices are obtained from ``pimple.config(fvSchemes)`` /
``pimple.config(fvSolution)`` so the declarations are scoped to this
model rather than mutating the shared base classes.
"""

from typing import Annotated, Any, Callable, Optional, Protocol

import pybFoam as pyf
from pybFoam import (
    fvc,
    fvm,
    fvScalarMatrix,
    fvVectorMatrix,
    surfaceScalarField,
    surfaceVectorField,
    volScalarField,
    volVectorField,
)

from neofoam import telemetry
from neofoam.fields import (
    CalculatedBC,
    CyclicBC,
    EmptyBC,
    FixedValueBC,
    GenericBC,
    InletOutletBC,
    NoSlipBC,
    PressureInletOutletVelocityBC,
    Scalar,
    SlipBC,
    SymmetryBC,
    SymmetryPlaneBC,
    Vector,
    ZeroGradientBC,
)
from neofoam.foam import fvSchemes, fvSolution
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.dependency_resolver import wrap_with_dependency_resolution
from neofoam.framework.initialization import field, model
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata

from ..incompressibleFluidModel import Model
from .control_factory import create_dynamic_mesh_controls, create_pimple_control

pimple = Model("Pimple")

# Per-spec fvSchemes / fvSolution slices. Operations extend these via
# ``@PimpleFvSchemes.add(...)`` / ``@PimpleFvSolution.add(...)`` below.
PimpleFvSchemes = pimple.config(fvSchemes)
PimpleFvSolution = pimple.config(fvSolution)

# Optional PIMPLE control keys read straight from ``system/fvSolution`` by
# ``setRefCell`` (see ``create_pressure_reference``). A closed domain (no
# fixed-pressure BC) needs a pressure reference; an open domain doesn't, so
# these stay optional and only serialise when the case author sets them.
PimpleFvSolution.add_controls("PIMPLE", pRefCell=int, pRefValue=float)

# 0/<name> field declarations PIMPLE owns. The framework auto-synthesises
# the matching read_field InitStep (see
# :func:`neofoam.fields.synthesis.synthesize_init_step`) and surfaces
# the schemas through ``configurations(solver).fields`` so the agent /
# case author fills the BCs without copying a source case. ``GenericBC``
# keeps unknown BC types (e.g. ``codedFixedValue``) parsing through the
# smart-union fallback.
pimple.field(
    "U",
    dimensions=[0, 1, -1, 0, 0, 0, 0],
    value_type=Vector,
    # Velocity-side arm set picked from the upstream-tutorial frequency
    # survey: noSlip / fixedValue dominate; pressureInletOutletVelocity,
    # slip, inletOutlet, and the topology arms (empty / symmetry* /
    # cyclic) round out >90% of all volVectorField patches.
    allowed_bcs=[
        NoSlipBC,
        FixedValueBC,
        ZeroGradientBC,
        SlipBC,
        InletOutletBC,
        PressureInletOutletVelocityBC,
        EmptyBC,
        SymmetryBC,
        SymmetryPlaneBC,
        CyclicBC,
        GenericBC,
    ],
    write=True,
)
pimple.field(
    "p",
    dimensions=[0, 2, -2, 0, 0, 0, 0],
    value_type=Scalar,
    # Pressure: fixedValue / zeroGradient / inletOutlet / calculated
    # dominate the upstream survey; topology arms cover thin / periodic
    # cases. ``GenericBC`` keeps the wall-functions (``totalPressure``,
    # adjoint pressure arms, …) parsing through the smart-union.
    allowed_bcs=[
        FixedValueBC,
        ZeroGradientBC,
        InletOutletBC,
        CalculatedBC,
        EmptyBC,
        SymmetryBC,
        SymmetryPlaneBC,
        CyclicBC,
        GenericBC,
    ],
    write=True,
)


class ViscousStress(Protocol):
    def update(self, ctx: Context) -> None: ...
    def divDevReff(self, U: volVectorField) -> Any: ...


def _report_continuity_errors(phi: surfaceScalarField, cumulativeContErr: list[float]) -> None:
    """Print the ``continuityErrs.H`` line and accumulate the global error."""
    sum_local, global_err = pyf.computeContinuityErrors(phi)
    cumulativeContErr[0] += global_err
    pyf.Info(
        f"time step continuity errors : sum local = {sum_local}, "
        f"global = {global_err}, cumulative = {cumulativeContErr[0]}"
    )


@pimple.build
def build(self: Any) -> list[Any]:
    """Lazy initializers for PIMPLE state (non-field bits only).

    ``U`` and ``p`` are auto-synthesized from the ``pimple.field(...)``
    declarations at the top of this module — the framework's
    :meth:`ModelRuntime.run_build` emits the matching read_field
    InitStep with ``depends_on`` and ``write`` flowing from the
    declaration. ``@build`` only carries what the framework cannot
    synthesize: ``phi`` (surfaceScalarField; computed from U), the
    pimpleControl object, the running continuity-error accumulator,
    and the pressure-reference cell logic. When ``use_boussinesq`` has
    been flipped (by the boussinesq plugin during resolve), the
    pressure-reference dependency list adds ``p_rgh`` and the
    reference cell logic adapts to use the modified-pressure field.
    """

    def create_phi(context: dict[str, Any]) -> surfaceScalarField:
        return pyf.createPhi(context["fields.U"])

    def create_cumulative_cont_err(_context: dict[str, Any]) -> list[float]:
        return [0.0]

    def create_Uf(context: dict[str, Any]) -> Optional[surfaceVectorField]:
        """Face velocity ``Uf`` — transcription of ``createUfIfPresent.H``.

        Only a dynamic mesh has one: it is what the moving-mesh
        ``fvc::ddtCorr(U, phi, Uf)`` and ``fvc::correctUf`` work on, and its
        absence is what tells the pressure corrector to keep the two-argument
        ``ddtCorr``. Native reads ``<time>/Uf`` when present (a restart); pybFoam
        has no READ_IF_PRESENT overload, so this always starts from
        ``fvc::interpolate(U)`` — identical for a run started from time 0.
        """
        if not context["mesh"].dynamic():
            return None
        pyf.Info("Constructing face velocity Uf")
        return surfaceVectorField(pyf.Word("Uf"), fvc.interpolate(context["fields.U"]))

    def create_pressure_reference(context: dict[str, Any]) -> dict[str, Any]:
        p = context["fields.p"]
        mesh = context["mesh"]
        use_boussinesq: bool = getattr(pimple, "use_boussinesq", False)
        p_rgh = context.get("fields.p_rgh") if use_boussinesq else None

        fv_solution = pyf.dictionary.read("system/fvSolution")
        algo_dict = fv_solution.subDict(pimple.algorithm_type)  # type: ignore[attr-defined]
        pressure_field = p_rgh if p_rgh is not None else p
        field_name = "p_rgh" if p_rgh is not None else "p"

        if not (
            algo_dict.found(f"{field_name}RefCell") or algo_dict.found(f"{field_name}RefPoint")
        ):
            if p_rgh is not None and (algo_dict.found("pRefCell") or algo_dict.found("pRefPoint")):
                pRefCell, pRefValue = pyf.setRefCell(p, algo_dict, True)
            else:
                pRefCell, pRefValue = pyf.setRefCell(pressure_field, algo_dict)
        else:
            pRefCell, pRefValue = pyf.setRefCell(pressure_field, algo_dict)

        mesh.setFluxRequired(pyf.Word("p"))
        if p_rgh is not None:
            mesh.setFluxRequired(pyf.Word("p_rgh"))

        return {"pRefCell": pRefCell, "pRefValue": pRefValue}

    init_steps = [
        # ``phi`` is a surfaceScalarField (no per-cell internalField /
        # boundaryField the way ``U`` / ``p`` carry, and no ``0/phi``
        # disk file in the standard sense) so it stays on the legacy
        # ``field()`` helper until a surface-field schema lands.
        field("phi", create_phi, depends_on=["fields.U"], write=True),
        model("pimple_control", create_pimple_control, depends_on=["mesh"]),
        model("dynamic_mesh_controls", create_dynamic_mesh_controls, depends_on=["mesh"]),
        model("Uf", create_Uf, depends_on=["mesh", "fields.U"]),
        model("cumulativeContErr", create_cumulative_cont_err),
    ]

    pressure_ref_deps = ["fields.p", "mesh"]
    if getattr(pimple, "use_boussinesq", False):
        pressure_ref_deps.append("fields.p_rgh")
    init_steps.append(
        model(
            "pressure_reference",
            create_pressure_reference,
            depends_on=pressure_ref_deps,
        )
    )

    return init_steps


def inner_loop(ctx: Context) -> bool:
    pimple = ctx.models["pimple_control"]
    looping = bool(pimple.loop())
    if looping:
        # Mirror pimpleControl::loop(): on the final outer iteration flag the
        # mesh so fvMatrix::solve picks the <field>Final solver settings (in
        # PISO mode, nOuterCorrectors 1, that is every iteration). The momentum
        # and pressure equations pass this flag explicitly (U.select / p.select
        # below), so the mesh flag is only needed for solves buried inside
        # OpenFOAM library code — the turbulence model's k/epsilon/nuTilda
        # solves in ``turbulence.correct()``, which take no argument, and
        # ``p.relax()``'s pFinal lookup. The flag deliberately stays raised on
        # exit (native lowers it there): with the correction now inside the
        # loop, only the write phase's function objects still see it, and they
        # match native today. The next step's first call lowers it.
        ctx.mesh.setFinalIteration(pimple.finalIter())
    return looping


@pimple.operation()
def mesh_update(
    ctx: Context,
    U: volVectorField,
    phi: surfaceScalarField,
    p: volScalarField,
    pimple_control: Annotated[Any, "models"],
    dynamic_mesh_controls: Annotated[dict[str, bool], "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    Uf: Annotated[Optional[surfaceVectorField], "models"] = None,
    mrf_zones: Annotated[Optional[pyf.IOMRFZoneList], "models"] = None,
) -> FieldUpdates:
    """Move the mesh at the head of the outer corrector (pimpleFoam's ``mesh.update()``).

    Runs on the first outer iteration only, unless the case sets
    ``moveMeshOuterCorrectors`` — the guard native uses. A mesh that actually
    changed invalidates the MRF zone faces (re-found on the new topology) and the
    face flux, which — when the case asks for ``correctPhi`` — is rebuilt from the
    mapped face velocity, projected divergence-free and handed on relative to the
    mesh motion.

    A static mesh returns immediately, so this operation cannot perturb a static
    case.
    """
    mesh = ctx.mesh
    if not mesh.dynamic():
        return FieldUpdates({})

    if not (pimple_control.firstIter() or dynamic_mesh_controls["moveMeshOuterCorrectors"]):
        return FieldUpdates({})

    # controlledUpdate(), not update(): the dynamicMeshDict may carry
    # updateControl/updateInterval, which move the mesh only every so many steps.
    mesh.controlledUpdateMesh()
    if not mesh.changing():
        return FieldUpdates({})

    if mrf_zones is not None:
        mrf_zones.update()
    if not dynamic_mesh_controls["correctPhi"]:
        return FieldUpdates({})

    assert Uf is not None  # createUfIfPresent.H gives every dynamic mesh a Uf
    # The mesh moved under the old flux, so phi no longer belongs to this geometry:
    # rebuild it as the absolute flux of the mapped face velocity, project that
    # divergence-free, and hand it on relative to the mesh motion. Unlike interFoam,
    # pimpleFoam's correctPhi.H always projects with a uniform rAUf of 1 rather than
    # the last corrector's 1/UEqn.A(), so no rAU has to survive the time step.
    phi.assign(mesh.Sf() & Uf)
    pyf.CorrectPhi(
        U,
        phi,
        p,
        pyf.dimensionedScalar(pyf.Word("rAUf"), pyf.dimTime, 1.0),
        pimple_control.nNonOrthogonalCorrectors,
    )
    _report_continuity_errors(phi, cumulativeContErr)
    fvc.makeRelative(phi, U)
    return FieldUpdates({"phi": phi})


@pimple.operation(operation_number="2.1")
@PimpleFvSchemes.add(
    ddt="ddt(U)",
    # ``div(phi,U)`` (convection) + the viscous-stress divergence emitted by
    # ``divDevReff(U)`` — both required for the momentum predictor.
    div=["div(phi,U)", "div((nuEff*dev2(T(grad(U)))))"],
    grad="grad(U)",
    laplacian="laplacian(nuEff,U)",
)
@PimpleFvSolution.add("U")
def momentum(
    U: volVectorField,
    phi: surfaceScalarField,
    p: volScalarField,
    viscousStress: Annotated[ViscousStress, "models"],
    pimple_control: Annotated[Any, "models"],
    ctx: Context,
    mrf_zones: Annotated[Optional[pyf.IOMRFZoneList], "models"] = None,
    fv_options: Annotated[Optional[pyf.fvOptions], "models"] = None,
) -> FieldUpdates:
    # Start-of-outer-iteration prevIter snapshot: ``pimpleControl::loop()`` calls
    # ``storePrevIterFields()`` so the pressure corrector's ``p.relax()`` has a
    # previous state to blend against; p is untouched until the pressure solve,
    # so storing here is equivalent (same placement as simpleAlgorithm).
    p.storePrevIter()

    # Refresh the effective viscosity right where it is consumed: the momentum
    # transport model owns nuEff (nu + nut); ``update`` reads the current nu/nut
    # from the Context (a laminar model has no nut, the OpenFOAM fallback owns its
    # own and no-ops here). nut is fixed across the PIMPLE outer iterations, so this
    # matches OpenFOAM's once-per-step eddy viscosity.
    with telemetry.span("momentum.assemble"):
        viscousStress.update(ctx)
        if mrf_zones is None:
            UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + viscousStress.divDevReff(U))
        else:
            # UEqn.H under a rotating frame: the wall velocities on the MRF
            # patches are set first (they feed the boundary coefficients of
            # ``div(phi,U)``), then the frame acceleration joins the sum.
            mrf_zones.correctBoundaryVelocity(U)
            UEqn = fvVectorMatrix(
                fvm.ddt(U) + fvm.div(phi, U) + mrf_zones.DDt(U) + viscousStress.divDevReff(U)
            )
        if fv_options is not None:
            # ``== fvOptions(U)`` moves the source to the right-hand side, i.e.
            # subtracts it from the assembled matrix. UEqn.H's three fvOptions
            # calls sit at three exact points, and the order is the physics: the
            # source joins the sum BEFORE relaxation, the constraints are applied
            # AFTER it, and the correction runs after the solve.
            UEqn = fvVectorMatrix(UEqn - fv_options(U))
        UEqn.relax()
        if fv_options is not None:
            fv_options.constrain(UEqn)

    if pimple_control.momentumPredictor():
        with telemetry.span("momentum.solve"):
            # Pass the final-iteration flag explicitly via ``U.select`` (so the
            # solve picks the UFinal settings) instead of relying on the mesh
            # finalIteration state. ``UEqn`` keeps its coefficients for the
            # pressure loop; the predictor system ``UEqn + grad(p)`` is a
            # separate matrix whose solve updates U.
            fvVectorMatrix(UEqn + fvc.grad(p)).solve(U.select(pimple_control.finalIter()))
        if fv_options is not None:
            fv_options.correct(U)

    return FieldUpdates({"UEqn": UEqn, "U": U})


@pimple.operation(operation_number="2.2", depends_on=["momentum"])
@PimpleFvSchemes.add(
    grad="grad(p)",
    laplacian="laplacian(rAU,p)",
    interpolation=["flux(HbyA)", "interpolate(rAU)", "dotInterpolate(S,U_0)"],
    snGrad="snGrad(p)",
)
@PimpleFvSolution.add("p")
def continuity(
    U: volVectorField,
    p: volScalarField,
    phi: surfaceScalarField,
    UEqn: fvVectorMatrix,
    pimple_control: Annotated[Any, "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    pressure_reference: Annotated[dict[str, Any], "models"],
    Uf: Annotated[Optional[surfaceVectorField], "models"] = None,
    mrf_zones: Annotated[Optional[pyf.IOMRFZoneList], "models"] = None,
    fv_options: Annotated[Optional[pyf.fvOptions], "models"] = None,
) -> FieldUpdates:
    """Pressure-velocity coupling.

    ``Uf`` is the moving-mesh face velocity — ``None`` on a static mesh, where
    every mesh-motion term below reduces to the static form it always had.
    """
    pRefCell = pressure_reference["pRefCell"]
    pRefValue = pressure_reference["pRefValue"]

    while pimple_control.correct():
        with telemetry.span("pressure.flux"):
            rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
            HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p))

            # ddtCorr: on a moving mesh the correction is built from the face
            # velocity instead of the flux — what native's fvc::ddtCorr(U, phi, Uf)
            # dispatches to when mesh.dynamic().
            ddt_corr = fvc.ddtCorr(U, phi) if Uf is None else fvc.ddtCorr(U, Uf)
            if mrf_zones is None:
                phiHbyA = surfaceScalarField(
                    pyf.Word("phiHbyA"),
                    fvc.flux(HbyA) + fvc.interpolate(rAU) * ddt_corr,
                )
            else:
                # pEqn.H under a rotating frame: the ddt correction is zeroed
                # inside the MRF cells (it belongs to the absolute frame), then
                # the whole flux is taken relative to the rotation.
                phiHbyA = surfaceScalarField(
                    pyf.Word("phiHbyA"),
                    fvc.flux(HbyA) + mrf_zones.zeroFilter(fvc.interpolate(rAU) * ddt_corr),
                )
                mrf_zones.makeRelative(phiHbyA)

            # adjustPhi balances the global flux, which only means anything
            # relative to the moving mesh — hence native's
            # makeRelative/makeAbsolute bracket. Both are no-ops on a static mesh,
            # but the pair is not bit-exact in floating point, so it stays behind
            # the same needReference() guard native uses (the guard adjustPhi
            # itself applies internally).
            needs_reference = p.needReference()
            if needs_reference:
                fvc.makeRelative(phiHbyA, U)
            pyf.adjustPhi(phiHbyA, U, p)
            if needs_reference:
                fvc.makeAbsolute(phiHbyA, U)
            if mrf_zones is None:
                pyf.constrainPressure(p, U, phiHbyA, rAU)
            else:
                pyf.constrainPressure(p, U, phiHbyA, rAU, mrf_zones)

        while pimple_control.correctNonOrthogonal():
            with telemetry.span("pressure.assemble"):
                pEqn = fvScalarMatrix(fvm.laplacian(rAU, p) - fvc.div(phiHbyA))
                pEqn.setReference(pRefCell, pRefValue, False)
            with telemetry.span("pressure.solve", final=pimple_control.finalInnerIter()):
                pEqn.solve(p.select(pimple_control.finalInnerIter()))

            if pimple_control.finalNonOrthogonalIter():
                phi.assign(phiHbyA - pEqn.flux())

        # Explicit pressure under-relaxation before the momentum corrector
        # (pEqn.H). A no-op unless the case declares a ``p`` field relaxation
        # factor; on the final outer iteration the name becomes ``pFinal``.
        p.relax()
        U.assign(HbyA - rAU * fvc.grad(p))
        U.correctBoundaryConditions()
        if fv_options is not None:
            # pEqn.H closes on a second ``fvOptions.correct(U)``: the corrector
            # has just overwritten U, so any correction the predictor applied is
            # gone.
            fv_options.correct(U)

        _report_continuity_errors(phi, cumulativeContErr)

        if Uf is not None:
            # Moving mesh: refresh the face velocity from the corrected U/phi and
            # hand phi on relative to the mesh motion (pEqn.H's last two lines).
            # They must stay after the continuity report: native reports on the
            # *absolute* flux the pressure solve produced, and this pair is what
            # turns it relative.
            fvc.correctUf(Uf, U, phi)
            fvc.makeRelative(phi, U)

    return FieldUpdates({"U": U, "p": p, "phi": phi})


@pimple.operation(operation_number="2.1")
@PimpleFvSchemes.add(
    ddt="ddt(U)",
    div=["div(phi,U)", "div((nuEff*dev2(T(grad(U)))))"],
    # ``grad(rhok)`` is needed by the ``corrected`` ``snGrad(rhok)`` below: the
    # non-orthogonal correction of ``fvc::snGrad(rhok)`` looks up the cell gradient.
    grad=["grad(U)", "grad(rhok)"],
    laplacian="laplacian(nuEff,U)",
    snGrad="snGrad(rhok)",
)
@PimpleFvSolution.add("U")
def momentum_boussinesq(
    U: volVectorField,
    phi: surfaceScalarField,
    p: volScalarField,
    viscousStress: Annotated[ViscousStress, "models"],
    pimple_control: Annotated[Any, "models"],
    p_rgh: volScalarField,
    rhok: volScalarField,
    ghf: surfaceScalarField,
    ctx: Context,
) -> FieldUpdates:
    mesh = U.mesh()

    # Refresh nuEff where it is consumed (see ``momentum``).
    with telemetry.span("momentum.assemble"):
        viscousStress.update(ctx)
        UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + viscousStress.divDevReff(U))
        UEqn.relax()

    if pimple_control.momentumPredictor():
        with telemetry.span("momentum.solve"):
            # Explicit final-iteration flag (see ``momentum``): pick UFinal via
            # U.select rather than the mesh finalIteration state.
            fvVectorMatrix(
                UEqn + fvc.reconstruct((-ghf * fvc.snGrad(rhok) - fvc.snGrad(p_rgh)) * mesh.magSf())
            ).solve(U.select(pimple_control.finalIter()))

    return FieldUpdates({"UEqn": UEqn, "U": U})


@pimple.operation(operation_number="2.2", depends_on=["momentum_boussinesq"])
@PimpleFvSchemes.add(
    grad="grad(p_rgh)",
    laplacian="laplacian(rAUf,p_rgh)",
    interpolation="flux(U)",
    snGrad="snGrad(p_rgh)",
)
@PimpleFvSolution.add("p_rgh")
def continuity_boussinesq(
    U: volVectorField,
    p: volScalarField,
    phi: surfaceScalarField,
    UEqn: fvVectorMatrix,
    pimple_control: Annotated[Any, "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    pressure_reference: Annotated[dict[str, Any], "models"],
    p_rgh: volScalarField,
    rhok: volScalarField,
    gh: volScalarField,
    ghf: surfaceScalarField,
) -> FieldUpdates:
    pRefCell = pressure_reference["pRefCell"]
    pRefValue = pressure_reference["pRefValue"]
    mesh = U.mesh()

    while pimple_control.correct():
        with telemetry.span("pressure.flux"):
            rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
            rAUf = surfaceScalarField(pyf.Word("rAUf"), fvc.interpolate(rAU))
            HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p_rgh))

            phig = surfaceScalarField(
                pyf.Word("phig"), -rAUf * ghf * fvc.snGrad(rhok) * mesh.magSf()
            )
            phiHbyA = surfaceScalarField(
                pyf.Word("phiHbyA"),
                fvc.flux(HbyA) + rAUf * fvc.ddtCorr(U, phi) + phig,
            )

            pyf.constrainPressure(p_rgh, U, phiHbyA, rAUf)

        while pimple_control.correctNonOrthogonal():
            with telemetry.span("pressure.assemble"):
                pEqn = fvScalarMatrix(fvm.laplacian(rAUf, p_rgh) - fvc.div(phiHbyA))
                pEqn.setReference(pRefCell, pRefValue, False)
            with telemetry.span("pressure.solve", final=pimple_control.finalInnerIter()):
                pEqn.solve(p_rgh.select(pimple_control.finalInnerIter()))

            if pimple_control.finalNonOrthogonalIter():
                phi.assign(phiHbyA - pEqn.flux())

        U.assign(HbyA + rAU * fvc.reconstruct((phig - pEqn.flux()) / rAUf))
        U.correctBoundaryConditions()
        p.assign(p_rgh + rhok * gh)

        sum_local, global_err = pyf.computeContinuityErrors(phi)
        cumulativeContErr[0] += global_err
        pyf.Info(
            f"time step continuity errors : sum local = {sum_local}, "
            f"global = {global_err}, cumulative = {cumulativeContErr[0]}"
        )

    return FieldUpdates({"U": U, "p": p, "phi": phi, "p_rgh": p_rgh})


def _alias_operation(
    op_func: Callable[..., Any],
    *,
    operation_name: str,
    depends_on: list[str],
) -> Operation:
    return Operation(
        func=SequentialOp(op_func),
        metadata=OperationMetadata(
            op_name=operation_name,
            depends_on=depends_on,
            shape="box",
            color="lightblue",
        ),
    )


@pimple.operation_collection
def collected_operations(self: Any) -> Operations:
    """Wrap momentum/continuity inside the PIMPLE inner loop.

    Dispatches to boussinesq variants when ``self.use_boussinesq`` is set
    (the boussinesq plugin flips that flag during its resolve stage).
    """
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=IterativeOp(inner_loop),
            metadata=OperationMetadata(op_name="inner_loop"),
        )
    )

    if getattr(self, "use_boussinesq", False):
        momentum_op = momentum_boussinesq
        continuity_op = continuity_boussinesq
    else:
        momentum_op = momentum
        continuity_op = continuity
        # Mesh motion is wired into the plain PIMPLE arm only. The boussinesq arm
        # carries a buoyancy head gh/ghf tied to the cell centres, which a mesh
        # move would invalidate; no buoyant tutorial has a dynamicMeshDict, so it
        # keeps the static path rather than gaining a silently stale one.
        model_ops.add(
            _alias_operation(
                wrap_with_dependency_resolution(mesh_update, self, pimple._dependency_resolver),
                operation_name="mesh_update",
                depends_on=[],
            )
        )

    wrapped_momentum = wrap_with_dependency_resolution(
        momentum_op, self, pimple._dependency_resolver
    )
    wrapped_continuity = wrap_with_dependency_resolution(
        continuity_op, self, pimple._dependency_resolver
    )

    model_ops.add(_alias_operation(wrapped_momentum, operation_name="momentum", depends_on=[]))
    model_ops.add(
        _alias_operation(
            wrapped_continuity,
            operation_name="continuity",
            depends_on=["momentum"],
        )
    )
    return model_ops
