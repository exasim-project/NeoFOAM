# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""PIMPLE algorithm for incompressibleVoF (interFoam-based) solver.

Implements:
  - momentum (density-weighted momentum with surface tension)
  - continuity (pressure-velocity coupling with gravity/surface tension)

Note: alpha_advection has been extracted into the AlphaAdvection core model
(see models/alpha_advection/alphaAdvectionModel.py).  This model owns fields:
  phi, mixture, alpha1, alpha2, rho, rhoPhi.
"""

import os
from pathlib import Path
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

from neofoam.algorithms.solution_loop.control import PimpleControl
from neofoam.fields import (
    CalculatedBC,
    CyclicBC,
    EmptyBC,
    FixedFluxPressureBC,
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
from neofoam.foam.initialization import read_vol_field
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization import field, model
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata

from ..alpha_advection.shared import MixtureProtocol
from ..incompressibleVoFModel import Model
from .control_factory import (
    FrozenFlowControl,
    create_dynamic_mesh_controls,
    create_pimple_control,
)
from .pressure_reference import get_ref_cell_value, need_reference, update_absolute_pressure

pimple = Model("Pimple")

# Per-spec fvSchemes / fvSolution slices — schema-only, surfaced through
# ``configurations(solver)`` so the case wizard / agent fills the case scaffold.
# Operations extend these via ``@PimpleFvSchemes.add(...)`` /
# ``@PimpleFvSolution.add(...)`` below. Declaring them does not change the solve:
# ``ModelSpec.build_steps`` runs only the ``@pimple.build`` function, never the
# field/config declarations, so the fields are still read by ``@build`` at run
# time exactly as before.
PimpleFvSchemes = pimple.config(fvSchemes)
PimpleFvSolution = pimple.config(fvSolution)

# Optional PIMPLE control keys read straight from ``system/fvSolution`` by
# ``setRefCell`` (see ``create_pressure_reference``): a closed domain (no
# fixed-pressure BC) needs a pressure reference, an open one does not.
PimpleFvSolution.add_controls("PIMPLE", pRefCell=int, pRefValue=float)

# ``pcorr`` is solved by the start-up flux projection (``initCorrectPhi.H``, see
# ``project_flux``), which runs for every case — so a case scaffold without a
# ``pcorr`` solver entry aborts on the first solve. Declared here rather than on an
# operation because the projection is an initialisation step, not a time-loop one.
PimpleFvSolution.add("pcorr")

# Section ``default`` schemes — interFoam's ``interfaceProperties`` resolves the
# interface-normal gradient through a ``nHat`` gradScheme and makes several other
# unnamed scheme lookups (curvature / surface-tension interpolation) that fall
# back to the section default. Without these, a wizard-authored VoF case (explicit
# entries only) aborts with *Entry 'nHat' not found in gradSchemes*. ``div`` is
# left out on purpose: interFoam keeps ``divSchemes default none`` so every
# convection term must be declared explicitly (they are, above).
PimpleFvSchemes.add(
    ddt="default",
    grad="default",
    laplacian="default",
    snGrad="default",
    interpolation="default",
)

# 0/<name> field declarations PIMPLE owns for the wizard: velocity ``U`` and the
# buoyant (dynamic) pressure ``p_rgh`` — the fields the case author fills BCs for.
# The absolute pressure ``p`` is derived (``p = p_rgh + rho*gh``) so it is not a
# separate authoring surface; ``alpha.water`` is owned by the advection family.
pimple.field(
    "U",
    dimensions=[0, 1, -1, 0, 0, 0, 0],
    value_type=Vector,
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
    "p_rgh",
    dimensions=[1, -1, -2, 0, 0, 0, 0],
    value_type=Scalar,
    # Buoyant pressure: fixedFluxPressure dominates on walls/tubes, fixedValue
    # at the outlet, inletOutlet/calculated round out the survey; topology arms
    # cover thin / periodic cases.
    allowed_bcs=[
        FixedFluxPressureBC,
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


# ---------------------------------------------------------------------------
# Type protocols for static analysis
# ---------------------------------------------------------------------------


class TwoPhaseTransportProtocol(Protocol):
    def divDevRhoReff(self, rho: volScalarField, U: volVectorField) -> fvVectorMatrix: ...
    def correct(self) -> None: ...


# ---------------------------------------------------------------------------
# Gravity head (gh / ghf) — built at init, rebuilt whenever the mesh moves
# ---------------------------------------------------------------------------


def _gravity_ref(mesh: Any, hRef: Any) -> tuple[Any, Any]:
    """Return ``(g, ghRef)`` — gravity field and its reference head.

    Transcription of ``gh.H``: ``ghRef = g & (cmptMag(g)/mag(g))*hRef``
    when ``|g| > SMALL``, else 0 — the reference head along the gravity
    direction, zero only when ``hRef`` is itself zero (the default).
    ``pyf.vector`` has no ``cmptMag``/``__truediv__`` binding, so the unit
    vector is built component-wise and by a ``1/mag(g)`` multiply instead.
    SMALL is OpenFOAM's ``doubleScalar`` constant (not exposed by
    pybFoam): 1e-15.
    """
    g = pyf.uniformDimensionedVectorField(mesh, "g")
    gh_ref_dims = g.dimensions() * pyf.dimLength
    g_val = g.value()
    mag_g = pyf.mag(g_val)
    if mag_g > 1e-15:
        cmpt_mag_g = pyf.vector(abs(g_val[0]), abs(g_val[1]), abs(g_val[2]))
        unit_g = cmpt_mag_g * (1.0 / mag_g)
        gh_ref_value = (g_val & unit_g) * hRef.value()
    else:
        gh_ref_value = 0.0
    return g, pyf.dimensionedScalar("ghRef", gh_ref_dims, gh_ref_value)


def update_gravity_head(gh: volScalarField, ghf: surfaceScalarField, mesh: Any, hRef: Any) -> None:
    """Re-evaluate ``gh``/``ghf`` on the current cell and face centres.

    The buoyancy head is tied to the mesh geometry, so interFoam rebuilds both
    fields inside its ``mesh.changing()`` block. Call it after every mesh move;
    on a static mesh the values never change and it is never needed.
    """
    g, gh_ref = _gravity_ref(mesh, hRef)
    gh.assign((g & mesh.C()) - gh_ref)
    ghf.assign((g & mesh.Cf()) - gh_ref)


# ---------------------------------------------------------------------------
# Flux projection (CorrectPhi) — start-up and after every mesh move
# ---------------------------------------------------------------------------


def _report_continuity_errors(phi: surfaceScalarField, cumulativeContErr: list[float]) -> None:
    """Print the ``continuityErrs.H`` line and accumulate the global error."""
    sum_local, global_err = pyf.computeContinuityErrors(phi)
    cumulativeContErr[0] += global_err
    pyf.Info(
        f"time step continuity errors : sum local = {sum_local}, "
        f"global = {global_err}, cumulative = {cumulativeContErr[0]}"
    )


def project_flux(
    U: volVectorField,
    phi: surfaceScalarField,
    p_rgh: volScalarField,
    rAU: Optional[volScalarField],
    pimple_control: Any,
    cumulativeContErr: list[float],
) -> None:
    """Project ``phi`` divergence-free (``CorrectPhi``) and report continuity.

    The shared body of ``initCorrectPhi.H`` and of interFoam's in-loop
    ``correctPhi.H``; both end in ``continuityErrs.H``, and they differ only in the
    ``rAUf`` they hand to ``CorrectPhi``.

    ``rAU is None`` selects the uniform ``dimensionedScalar`` 1 of
    ``initCorrectPhi.H``'s ``else`` branch. Its ``correctPhi`` branch instead
    interpolates a *uniform-1* ``rAU`` field, and ``fvm::laplacian`` expands a
    dimensioned scalar into exactly that uniform surface field under the same
    ``rAUf`` name — so the two start-up branches assemble the same matrix and one
    code path covers both. Once the pressure corrector has run, ``rAU`` is its
    ``1/UEqn.A()``, re-interpolated here on the *current* (moved) mesh, which is
    what ``correctPhi.H`` does after ``mesh.update()``.

    The non-orthogonal corrector count comes from the control rather than a
    ``pimpleControl``: it is the only thing native's ``CorrectPhi`` asks the
    control for.
    """
    if rAU is None:
        rAUf: Any = pyf.dimensionedScalar(pyf.Word("rAUf"), pyf.dimTime / pyf.dimDensity, 1.0)
    else:
        rAUf = surfaceScalarField(pyf.Word("rAUf"), fvc.interpolate(rAU))

    pyf.CorrectPhi(U, phi, p_rgh, rAUf, pimple_control.nNonOrthogonalCorrectors)
    _report_continuity_errors(phi, cumulativeContErr)


# ---------------------------------------------------------------------------
# Build: register initialisation steps provided by pimple model
# ---------------------------------------------------------------------------


@pimple.build
def build(self: object) -> list[object]:  # noqa: C901
    """Register all field and model initialisation steps for VoF PIMPLE algorithm."""

    # ------------------------------------------------------------------ #
    # Factory functions (closures) for each field / model                 #
    # ------------------------------------------------------------------ #

    def create_hRef(context: dict[str, Any]) -> Any:
        """Read ``hRef`` from ``constant/`` (transcription of ``readhRef.H``).

        Native: ``uniformDimensionedScalarField hRef(IOobject("hRef", ...,
        READ_IF_PRESENT, NO_WRITE), dimensionedScalar(dimLength, Zero))`` — a
        case with no ``constant/hRef`` silently gets ``hRef = 0``, and either
        way the object is registered so
        ``prghPermeableAlphaTotalPressure``-style BCs can
        ``lookupObject<uniformDimensionedScalarField>("hRef")`` on every
        evaluation, for the whole run.

        pybFoam's ``uniformDimensionedScalarField`` binding is hard-coded
        ``MUST_READ`` (no ``READ_IF_PRESENT``/default-value overload), and a
        missing file raises an OpenFOAM ``FatalError`` that calls
        ``std::exit`` directly rather than a Python-catchable exception, so a
        plain ``try/except`` around it cannot reproduce the fallback. When
        ``constant/hRef`` is absent, this writes the same default OpenFOAM
        would use in memory (``dimensions [0 1 0 0 0 0 0]; value 0;``) so the
        ``MUST_READ`` read then succeeds and returns a properly registered
        field — the only way pybFoam can construct one with a value.
        """
        mesh = context["mesh"]
        href_path = Path("constant/hRef")
        if not href_path.exists():
            # constant/ is shared by every MPI rank; write via a rank-unique temp
            # file + atomic rename so no rank can read a partially written hRef.
            tmp_path = href_path.with_name(f".hRef.{os.getpid()}.tmp")
            tmp_path.write_text(
                "FoamFile\n"
                "{\n"
                "    version     2.0;\n"
                "    format      ascii;\n"
                "    class       uniformDimensionedScalarField;\n"
                "    object      hRef;\n"
                "}\n"
                "\n"
                "dimensions      [0 1 0 0 0 0 0];\n"
                "value           0;\n"
            )
            tmp_path.replace(href_path)
        return pyf.uniformDimensionedScalarField(mesh, "hRef")

    def create_gh(context: dict[str, Any]) -> volScalarField:
        mesh = context["mesh"]
        g, gh_ref_dim = _gravity_ref(mesh, context["fields.hRef"])
        return volScalarField(pyf.Word("gh"), (g & mesh.C()) - gh_ref_dim)

    def create_ghf(context: dict[str, Any]) -> surfaceScalarField:
        mesh = context["mesh"]
        g, gh_ref_dim = _gravity_ref(mesh, context["fields.hRef"])
        return surfaceScalarField(pyf.Word("ghf"), (g & mesh.Cf()) - gh_ref_dim)

    def create_p(context: dict[str, Any]) -> volScalarField:
        p_rgh = context["fields.p_rgh"]
        rho = context["fields.rho"]
        gh = context["fields.gh"]
        # Absolute pressure: p = p_rgh + rho*gh
        return volScalarField(pyf.Word("p"), p_rgh + rho * gh)

    def create_p_rgh(context: dict[str, Any]) -> volScalarField:
        mesh = context["mesh"]
        mesh.setFluxRequired(pyf.Word("p_rgh"))
        return volScalarField.read_field(mesh, "p_rgh")

    def create_Uf(context: dict[str, Any]) -> Optional[surfaceVectorField]:
        """Face velocity ``Uf`` — transcription of ``createUfIfPresent.H``.

        Only a dynamic mesh has one: it is what the moving-mesh
        ``fvc::ddtCorr(U, phi, Uf)`` and ``fvc::correctUf`` work on. ``None`` on a
        static mesh, which is what tells the pressure corrector to keep the
        two-argument ``ddtCorr``. Native reads ``<time>/Uf`` when present (a
        restart); pybFoam has no READ_IF_PRESENT overload, so this always starts
        from ``fvc::interpolate(U)`` — identical for a run started from time 0.
        """
        mesh = context["mesh"]
        if not mesh.dynamic():
            return None
        pyf.Info("Constructing face velocity Uf")
        return surfaceVectorField(pyf.Word("Uf"), fvc.interpolate(context["fields.U"]))

    def create_pressure_reference(context: dict[str, Any]) -> dict[str, Any]:
        """Set reference cell/value for p_rgh and level p/p_rgh against it."""
        p = context["fields.p"]
        p_rgh = context["fields.p_rgh"]
        fv_solution = pyf.dictionary.read("system/fvSolution")
        algo_dict = fv_solution.subDict("PIMPLE")
        # Two-field setRefCell(p, p_rgh, dict) overload. OpenFOAM's setRefCell
        # returns whether a reference is needed at all, but the pybFoam binding
        # drops that bool — ``need_reference`` recovers it from the cell index,
        # which setRefCell sets negative on the no-reference path. Must stay
        # ``forceReference=False`` for that sentinel to hold; see its docstring.
        pRefCell, pRefValue = pyf.setRefCell(p, p_rgh, algo_dict, False)
        needsRef = need_reference(pRefCell)
        if needsRef:
            # createFields.H levels both fields against the reference cell right
            # after setRefCell, not only inside the corrector. A frozen-flow case
            # never runs the corrector, so without this its p_rgh is written back
            # at the level ``0/p_rgh`` happened to carry.
            update_absolute_pressure(
                p,
                p_rgh,
                context["fields.rho"],
                context["fields.gh"],
                ref_cell=pRefCell,
                ref_value=pRefValue,
                needs_reference=True,
            )
        return {
            "pRefCell": pRefCell,
            "pRefValue": pRefValue,
            "needsRef": needsRef,
        }

    def create_initial_flux_correction(context: dict[str, Any]) -> None:
        """Project the start-up flux — ``initCorrectPhi.H``, run unconditionally.

        interFoam includes this before the time loop with no enclosing ``if``: the
        ``correctPhi`` key only picks which of two equivalent ``rAUf`` forms is
        passed (and whether ``rAU`` survives into ``pEqn.H``), never *whether* the
        projection happens. Cases that leave the key unset need it just as much —
        ``createPhi(U)`` is only divergence-free if ``0/U`` already was, and the
        first Courant number, time step and alpha solve all read the projected flux.
        On a case whose initial flux does close, ``pcorr`` converges at iteration 0
        and ``phi`` comes back unchanged.
        """
        project_flux(
            context["fields.U"],
            context["fields.phi"],
            context["fields.p_rgh"],
            None,
            context["models.pimple_control"],
            context["models.cumulativeContErr"],
        )

    # ------------------------------------------------------------------ #
    # Init step list (dependency-ordered)                                 #
    # ------------------------------------------------------------------ #

    return [
        read_vol_field(volVectorField, "U"),
        field("p_rgh", create_p_rgh, depends_on=["mesh"]),
        field("hRef", create_hRef, depends_on=["mesh"]),
        field("gh", create_gh, depends_on=["mesh", "fields.hRef"]),
        field("ghf", create_ghf, depends_on=["mesh", "fields.hRef"]),
        field(
            "p",
            create_p,
            depends_on=["fields.p_rgh", "fields.rho", "fields.gh"],
        ),
        model("pimple_control", create_pimple_control, depends_on=["mesh"]),
        model(
            "dynamic_mesh_controls",
            create_dynamic_mesh_controls,
            depends_on=["mesh"],
        ),
        model("Uf", create_Uf, depends_on=["mesh", "fields.U"]),
        model("cumulativeContErr", lambda _: [0.0]),
        # interFoam keeps `rAU` alive across time steps when `correctPhi` is set,
        # because the post-move projection re-uses the last corrector's
        # `1/UEqn.A()` (pEqn.H:2-9). One-element box, written by `continuity`;
        # `None` until the first pressure solve, which is the uniform-1 `rAU`
        # initCorrectPhi.H starts from.
        model("last_rAU", lambda _: [None]),
        model(
            "pressure_reference",
            create_pressure_reference,
            depends_on=["fields.p", "fields.p_rgh", "fields.rho", "fields.gh", "mesh"],
        ),
        model(
            "initial_flux_correction",
            create_initial_flux_correction,
            depends_on=[
                "fields.U",
                "fields.phi",
                "fields.p_rgh",
                # createFields.H builds rhoPhi from the *un*projected phi, before
                # initCorrectPhi.H — depend on it so that order is preserved.
                "fields.rhoPhi",
                "models.pimple_control",
                "models.cumulativeContErr",
                "models.pressure_reference",
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Inner loop predicate
# ---------------------------------------------------------------------------


def inner_loop(ctx: Context) -> bool:
    pimple_control = ctx.models["pimple_control"]
    looping = bool(pimple_control.loop(ctx))
    # Mirror pimpleControl::loop() (pimpleControl.C): on the final outer iteration
    # flag the mesh so fvMatrix::solve picks the <field>Final settings for the
    # library solves buried in turbulence.correct() (k/epsilon/nuTilda), and lower
    # the flag again when the loop ends (native calls setFinalIteration(false)
    # before returning false). ``pimple_control.loop`` resets its counters on exit,
    # so ``finalIter()`` is then False. Here turbulence_correction runs *inside*
    # the inner loop, so lowering on exit means write_output's equation-solving
    # function objects (e.g. electricPotential) see the base solver dict, not the
    # non-existent <field>Final variant.
    ctx.mesh.setFinalIteration(pimple_control.finalIter())
    return looping


# ---------------------------------------------------------------------------
# Helper: alias a registered operation under a different name
# ---------------------------------------------------------------------------


def _alias_operation(
    op_func: Callable[..., FieldUpdates],
    *,
    operation_name: str,
    depends_on: list[str],
) -> Operation:
    return Operation(
        func=SequentialOp(op_func),
        metadata=OperationMetadata(
            op_name=operation_name,
            depends_on=depends_on,
            before=[],
            shape="box",
            color="lightblue",
        ),
    )


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@pimple.operation()
def mesh_update(
    ctx: Context,
    hRef: Any,
    gh: volScalarField,
    ghf: surfaceScalarField,
    U: volVectorField,
    phi: surfaceScalarField,
    p_rgh: volScalarField,
    pimple_control: Annotated[Any, "models"],
    dynamic_mesh_controls: Annotated[dict[str, bool], "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    last_rAU: Annotated[list[Optional[volScalarField]], "models"],
    mixture: Annotated[MixtureProtocol, "models"],
    Uf: Annotated[Optional[surfaceVectorField], "models"] = None,
    mrf_zones: Annotated[Optional[pyf.IOMRFZoneList], "models"] = None,
) -> FieldUpdates:
    """Move the mesh at the head of the outer corrector (interFoam's ``mesh.update()``).

    Runs on the first outer iteration only, unless the case sets
    ``moveMeshOuterCorrectors`` — the guard native uses. A mesh that actually
    changed invalidates the buoyancy head, so ``gh``/``ghf`` are rebuilt on the
    new cell/face centres, the MRF zone faces are re-found on the new topology,
    and — when the case asks for ``correctPhi`` — the flux is rebuilt from the
    mapped face velocity and re-projected onto the moved geometry.

    A static mesh returns immediately, so this operation cannot perturb a static
    case.
    """
    mesh = ctx.mesh
    if not mesh.dynamic():
        return FieldUpdates({})

    if not (pimple_control.firstIter() or dynamic_mesh_controls["moveMeshOuterCorrectors"]):
        return FieldUpdates({})

    mesh.updateMesh()
    if not mesh.changing():
        return FieldUpdates({})

    update_gravity_head(gh, ghf, mesh, hRef)
    if mrf_zones is not None:
        mrf_zones.update()
    if not dynamic_mesh_controls["correctPhi"]:
        return FieldUpdates({"gh": gh, "ghf": ghf})

    assert Uf is not None  # createUfIfPresent.H gives every dynamic mesh a Uf
    # The mesh moved under the old flux, so phi no longer belongs to this geometry:
    # rebuild it as the absolute flux of the mapped face velocity, project that
    # divergence-free, and hand it on relative to the mesh motion. mixture.correct()
    # then re-evaluates the interface properties on the corrected flux.
    phi.assign(mesh.Sf() & Uf)
    project_flux(U, phi, p_rgh, last_rAU[0], pimple_control, cumulativeContErr)
    fvc.makeRelative(phi, U)
    mixture.correct()
    return FieldUpdates({"gh": gh, "ghf": ghf, "phi": phi})


@pimple.operation(operation_number="2.1")
@PimpleFvSchemes.add(
    ddt="ddt(U)",
    # density-weighted convection + the viscous-stress divergence from
    # ``divDevRhoReff(rho, U)``.
    div=["div(rhoPhi,U)", "div(((rho*nuEff)*dev2(T(grad(U)))))"],
    grad="grad(U)",
    laplacian="laplacian(nuEff,U)",
    snGrad=["snGrad(rho)", "snGrad(p_rgh)"],
)
@PimpleFvSolution.add("U")
def momentum(
    U: volVectorField,
    rho: volScalarField,
    rhoPhi: surfaceScalarField,
    p_rgh: volScalarField,
    gh: volScalarField,
    ghf: surfaceScalarField,
    pimple_control: Annotated[PimpleControl, "models"],
    mixture: Annotated[MixtureProtocol, "models"],
    turbulence: Annotated[TwoPhaseTransportProtocol, "models"],
    mrf_zones: Annotated[Optional[pyf.IOMRFZoneList], "models"] = None,
    fv_options: Annotated[Optional[pyf.fvOptions], "models"] = None,
) -> FieldUpdates:
    """Density-weighted momentum predictor with surface tension."""
    if isinstance(pimple_control, FrozenFlowControl):
        # frozenFlow yes ⇒ interIsoFoam runs `continue` before UEqn.H: the
        # momentum equation is never assembled. Skip it entirely so the frozen
        # tutorials' deliberate omission of the momentum divSchemes
        # (e.g. div(rhoPhi,U)) is honoured rather than a fatal lookup.
        return FieldUpdates({})

    mesh = U.mesh()

    if mrf_zones is None:
        UEqn = fvVectorMatrix(
            fvm.ddt(rho, U) + fvm.div(rhoPhi, U) + turbulence.divDevRhoReff(rho, U)
        )
    else:
        # UEqn.H under a rotating frame: the wall velocities on the MRF patches
        # are set first (they feed the boundary coefficients of ``div(rhoPhi,U)``),
        # then the mass-weighted frame acceleration joins the sum.
        mrf_zones.correctBoundaryVelocity(U)
        UEqn = fvVectorMatrix(
            fvm.ddt(rho, U)
            + fvm.div(rhoPhi, U)
            + mrf_zones.DDt(rho, U)
            + turbulence.divDevRhoReff(rho, U)
        )
    if fv_options is not None:
        # ``== fvOptions(rho, U)`` moves the mass-weighted source to the
        # right-hand side, i.e. subtracts it from the assembled matrix. UEqn.H's
        # three fvOptions calls sit at three exact points, and the order is the
        # physics: the source joins the sum BEFORE relaxation, the constraints
        # are applied AFTER it, and the correction runs after the solve.
        UEqn = fvVectorMatrix(UEqn - fv_options(rho, U))
    UEqn.relax()
    if fv_options is not None:
        fv_options.constrain(UEqn)

    if pimple_control.momentumPredictor():
        pyf.solve(
            UEqn
            + fvc.reconstruct(
                (mixture.surfaceTensionForce() - ghf * fvc.snGrad(rho) - fvc.snGrad(p_rgh))
                * mesh.magSf()
            )
        )
        if fv_options is not None:
            fv_options.correct(U)

    return FieldUpdates({"UEqn": UEqn, "U": U})


@pimple.operation(operation_number="2.2", depends_on=["momentum"])
@PimpleFvSchemes.add(
    grad="grad(p_rgh)",
    laplacian="laplacian(rAUf,p_rgh)",
    # ``flux(U)`` builds the initial face flux phi (``createPhi(U)`` in the
    # advection shared build); OpenFOAM looks it up in interpolationSchemes, so it
    # must be declared or the case aborts *Entry 'flux(U)' not found*.
    interpolation=["flux(U)", "flux(HbyA)", "interpolate(rho*rAU)"],
    snGrad=["snGrad(p_rgh)", "snGrad(rho)"],
)
@PimpleFvSolution.add("p_rgh")
def continuity(
    U: volVectorField,
    rho: volScalarField,
    p_rgh: volScalarField,
    p: volScalarField,
    phi: surfaceScalarField,
    gh: volScalarField,
    ghf: surfaceScalarField,
    pimple_control: Annotated[PimpleControl, "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    pressure_reference: Annotated[dict[str, Any], "models"],
    mixture: Annotated[MixtureProtocol, "models"],
    last_rAU: Annotated[list[Optional[volScalarField]], "models"],
    Uf: Annotated[Optional[surfaceVectorField], "models"] = None,
    mrf_zones: Annotated[Optional[pyf.IOMRFZoneList], "models"] = None,
    fv_options: Annotated[Optional[pyf.fvOptions], "models"] = None,
    # Default None so a frozen-flow step (where ``momentum`` produced no UEqn)
    # still binds; the non-frozen path always supplies it.
    UEqn: Optional[fvVectorMatrix] = None,
) -> FieldUpdates:
    """Pressure-velocity coupling for VoF with surface tension and gravity.

    ``Uf`` is the moving-mesh face velocity — ``None`` on a static mesh, where
    every mesh-motion term below reduces to the static form it always had.
    """
    if isinstance(pimple_control, FrozenFlowControl):
        # frozenFlow yes ⇒ the pressure-corrector loop never runs (mirrors
        # interIsoFoam's `continue`): U/p/p_rgh/phi keep their prescribed values
        # and only alpha advects. Emit no field updates.
        return FieldUpdates({})
    assert UEqn is not None  # non-frozen path always carries the momentum matrix
    pRefCell = pressure_reference["pRefCell"]
    pRefValue = pressure_reference["pRefValue"]
    needsRef = pressure_reference["needsRef"]
    mesh = U.mesh()

    while pimple_control.correct():
        rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
        # Hand it to the post-move projection, which re-interpolates it on the
        # moved mesh next time step (interFoam keeps `rAU` alive for exactly this).
        last_rAU[0] = rAU
        rAUf = surfaceScalarField(pyf.Word("rAUf"), fvc.interpolate(rAU))

        HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p_rgh))

        # ddtCorr: on a moving mesh the correction is built from the face velocity
        # instead of the flux — what native's fvc::ddtCorr(U, phi, Uf) dispatches
        # to when mesh.dynamic().
        ddt_corr = fvc.ddtCorr(U, phi) if Uf is None else fvc.ddtCorr(U, Uf)
        if mrf_zones is None:
            phiHbyA = surfaceScalarField(
                pyf.Word("phiHbyA"),
                fvc.flux(HbyA) + fvc.interpolate(rho * rAU) * ddt_corr,
            )
        else:
            # pEqn.H under a rotating frame: the ddt correction is zeroed inside
            # the MRF cells (it belongs to the absolute frame), then the whole
            # flux is taken relative to the rotation.
            phiHbyA = surfaceScalarField(
                pyf.Word("phiHbyA"),
                fvc.flux(HbyA) + mrf_zones.zeroFilter(fvc.interpolate(rho * rAU) * ddt_corr),
            )
            mrf_zones.makeRelative(phiHbyA)

        # Surface tension + gravity contribution on faces
        phig = surfaceScalarField(
            pyf.Word("phig"),
            (mixture.surfaceTensionForce() - ghf * fvc.snGrad(rho)) * rAUf * mesh.magSf(),
        )
        phiHbyA.assign(phiHbyA + phig)

        # adjustPhi balances the global flux, which only means anything relative to
        # the moving mesh — hence native's makeRelative/makeAbsolute bracket. Both
        # are no-ops on a static mesh, but the pair is not bit-exact in floating
        # point, so it stays behind the same needReference() guard native uses.
        if needsRef:
            fvc.makeRelative(phiHbyA, U)
        pyf.adjustPhi(phiHbyA, U, p_rgh)
        if needsRef:
            fvc.makeAbsolute(phiHbyA, U)

        if mrf_zones is None:
            pyf.constrainPressure(p_rgh, U, phiHbyA, rAUf)
        else:
            pyf.constrainPressure(p_rgh, U, phiHbyA, rAUf, mrf_zones)

        while pimple_control.correctNonOrthogonal():
            pEqn = fvScalarMatrix(fvm.laplacian(rAUf, p_rgh) - fvc.div(phiHbyA))
            # pEqn.H pins the matrix at p_rgh's *current* level, not at pRefValue
            # (which is a level for the absolute pressure p). Using pRefValue here
            # would shift the whole p_rgh solution between the solve and the
            # re-levelling below, changing the solver's residual normalisation.
            pEqn.setReference(pRefCell, get_ref_cell_value(p_rgh, pRefCell), False)
            pEqn.solve(p_rgh.select(pimple_control.finalInnerIter()))

            if pimple_control.finalNonOrthogonalIter():
                phi.assign(phiHbyA - pEqn.flux())

        # Reconstruct velocity with surface tension + pressure flux correction.
        # pEqn deliberately leaks out of the non-orthogonal while loop above —
        # correctNonOrthogonal() always runs at least once, so pEqn is the
        # final-iteration matrix here (mirrors pEqn.H's scoping).
        U.assign(HbyA + rAU * fvc.reconstruct((phig - pEqn.flux()) / rAUf))
        U.correctBoundaryConditions()
        if fv_options is not None:
            # pEqn.H closes the corrector on a second ``fvOptions.correct(U)``:
            # the reconstruction has just overwritten U, so any correction the
            # predictor applied is gone.
            fv_options.correct(U)

        # Update absolute pressure from dynamic pressure; on a closed domain
        # also pin its level to pRefValue and re-level p_rgh (interFoam pEqn.H).
        update_absolute_pressure(
            p,
            p_rgh,
            rho,
            gh,
            ref_cell=pRefCell,
            ref_value=pRefValue,
            needs_reference=needsRef,
        )

        _report_continuity_errors(phi, cumulativeContErr)

        if Uf is not None:
            # Moving mesh: refresh the face velocity from the corrected U/phi and
            # hand phi on relative to the mesh motion (interFoam pEqn.H). Native
            # does both right after the continuity errors, which is where they are
            # here; the pressure update in between reads neither phi nor Uf.
            fvc.correctUf(Uf, U, phi)
            fvc.makeRelative(phi, U)

    return FieldUpdates({"U": U, "p": p, "p_rgh": p_rgh, "phi": phi})


# ---------------------------------------------------------------------------
# Operation collection: expose inner_loop + momentum/continuity aliases
# ---------------------------------------------------------------------------


@pimple.operation_collection
def collected_operations(self: object) -> Operations:
    """Wrap momentum/continuity inside the PIMPLE inner loop.

    The operation-collection path bypasses the spec's default operation
    wrapping, so momentum/continuity are wrapped with dependency resolution
    here (``self`` is the bound runtime) — mirroring the incompressibleFluid
    PIMPLE model.
    """
    model_ops = Operations()

    model_ops.add(
        Operation(
            func=IterativeOp(inner_loop),
            metadata=OperationMetadata(op_name="inner_loop"),
        )
    )

    wrapped_mesh_update = pimple.wrap_operation(mesh_update, self)
    wrapped_momentum = pimple.wrap_operation(momentum, self)
    wrapped_continuity = pimple.wrap_operation(continuity, self)

    model_ops.add(
        _alias_operation(wrapped_mesh_update, operation_name="mesh_update", depends_on=[])
    )
    model_ops.add(_alias_operation(wrapped_momentum, operation_name="momentum", depends_on=[]))
    model_ops.add(
        _alias_operation(wrapped_continuity, operation_name="continuity", depends_on=["momentum"])
    )

    return model_ops
