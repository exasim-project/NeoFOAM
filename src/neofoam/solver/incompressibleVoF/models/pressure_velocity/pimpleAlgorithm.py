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

from neofoam.algorithms import PressureReference, correct_phi
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
from neofoam.foam.algorithm_configs import DynamicMeshControls
from neofoam.foam.initialization import read_vol_field
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization import field, model
from neofoam.framework.model import BoundExtension
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
    VofPimpleAlgorithmConfig,
    create_dynamic_mesh_controls,
    create_pimple_control,
)
from .extension import (
    mesh_update_extension,
    momentum_extension,
    pressure_extension,
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

# Declared so ``configurations(solver)`` and the MCP export the key set.
# Loading is unaffected: this spec is used as its own runtime and never
# auto-loads its configs — ``control_factory`` drives instantiation.
pimple.config(VofPimpleAlgorithmConfig)
pimple.config(DynamicMeshControls)

# ``pcorr`` is solved by the start-up flux projection (``initCorrectPhi.H``), which
# runs for every case. Declared on the model rather than on an operation because the
# projection is an initialisation step, not a time-loop one.
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
    """Return ``(g, ghRef)`` — gravity field and its reference head (``gh.H``)."""
    g = pyf.uniformDimensionedVectorField(mesh, "g")
    gh_ref_dims = g.dimensions() * pyf.dimLength
    g_val = g.value()
    mag_g = pyf.mag(g_val)
    # 1e-15 is OpenFOAM's SMALL, which pybFoam does not expose. ``pyf.vector`` has
    # no ``cmptMag``/``__truediv__``, hence the component-wise unit vector below.
    if mag_g > 1e-15:
        cmpt_mag_g = pyf.vector(abs(g_val[0]), abs(g_val[1]), abs(g_val[2]))
        unit_g = cmpt_mag_g * (1.0 / mag_g)
        gh_ref_value = (g_val & unit_g) * hRef.value()
    else:
        gh_ref_value = 0.0
    return g, pyf.dimensionedScalar("ghRef", gh_ref_dims, gh_ref_value)


def update_gravity_head(gh: volScalarField, ghf: surfaceScalarField, mesh: Any, hRef: Any) -> None:
    """Re-evaluate ``gh``/``ghf`` on the current cell and face centres.

    The buoyancy head is tied to the mesh geometry, so call this after every mesh
    move (interFoam's ``mesh.changing()`` block); a static mesh never needs it.
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

    The shared body of ``initCorrectPhi.H`` and interFoam's in-loop
    ``correctPhi.H``, which differ only in the ``rAUf`` handed to ``CorrectPhi``:
    ``rAU is None`` is the start-up uniform 1, otherwise it is the last
    corrector's ``1/UEqn.A()`` re-interpolated on the current (moved) mesh.
    """
    if rAU is None:
        rAUf: Any = pyf.dimensionedScalar(pyf.Word("rAUf"), pyf.dimTime / pyf.dimDensity, 1.0)
    else:
        rAUf = surfaceScalarField(pyf.Word("rAUf"), fvc.interpolate(rAU))

    correct_phi(U, phi, p_rgh, rAUf, pimple_control.nNonOrthogonalCorrectors)
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
        """Read ``hRef`` from ``constant/`` (``readhRef.H``), defaulting it to 0."""
        # pybFoam's binding is hard-coded MUST_READ (no READ_IF_PRESENT overload)
        # and a missing file exits the process rather than raising, so native's
        # default is reproduced by writing the file OpenFOAM would have defaulted to.
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
        """Face velocity ``Uf`` (createUfIfPresent.H); ``None`` selects the static ``ddtCorr``."""
        # pybFoam has no READ_IF_PRESENT overload, so a restart cannot read
        # ``<time>/Uf`` and always starts from ``fvc::interpolate(U)``.
        mesh = context["mesh"]
        if not mesh.dynamic():
            return None
        pyf.Info("Constructing face velocity Uf")
        return surfaceVectorField(pyf.Word("Uf"), fvc.interpolate(context["fields.U"]))

    def create_pressure_reference(context: dict[str, Any]) -> PressureReference:
        """Set reference cell/value for p_rgh and level p/p_rgh against it."""
        p = context["fields.p"]
        p_rgh = context["fields.p_rgh"]
        fv_solution = pyf.dictionary.read("system/fvSolution")
        algo_dict = fv_solution.subDict("PIMPLE")
        # Two-field setRefCell(p, p_rgh, dict) overload. The pybFoam binding drops
        # native's "reference needed" return value, so ``need_reference`` recovers it
        # from the negative cell index — which holds only for ``forceReference=False``.
        pRefCell, pRefValue = pyf.setRefCell(p, p_rgh, algo_dict, False)
        needsRef = need_reference(pRefCell)
        if needsRef:
            # as in createFields.H: level both fields right after setRefCell, not
            # only inside the corrector, which a frozen-flow case never runs.
            update_absolute_pressure(
                p,
                p_rgh,
                context["fields.rho"],
                context["fields.gh"],
                ref_cell=pRefCell,
                ref_value=pRefValue,
                needs_reference=True,
            )
        return PressureReference(cell=pRefCell, value=pRefValue, needs_ref=needsRef)

    def create_initial_flux_correction(context: dict[str, Any]) -> None:
        """Project the start-up flux — ``initCorrectPhi.H``, run unconditionally.

        interFoam includes this with no enclosing ``if``: the ``correctPhi`` key
        picks which ``rAUf`` form is used, never whether the projection happens.
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
        # interFoam keeps `rAU` alive across time steps when `correctPhi` is set: the
        # post-move projection re-uses the last corrector's `1/UEqn.A()` (pEqn.H:2-9).
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
    # Mirror pimpleControl::loop(): on the final outer iteration flag the mesh so
    # fvMatrix::solve picks the <field>Final settings for the library solves buried
    # in turbulence.correct(). Set unconditionally, so the flag is also lowered on
    # exit (as native does) and write_output's function objects see the base dict.
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
    dynamic_mesh_controls: Annotated[DynamicMeshControls, "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    last_rAU: Annotated[list[Optional[volScalarField]], "models"],
    mixture: Annotated[MixtureProtocol, "models"],
    ext: Annotated[BoundExtension, mesh_update_extension],
    Uf: Annotated[Optional[surfaceVectorField], "models"] = None,
) -> FieldUpdates:
    """Move the mesh at the head of the outer corrector (interFoam's ``mesh.update()``).

    A static mesh returns immediately, so this operation cannot perturb a static case.
    """
    mesh = ctx.mesh
    if not mesh.dynamic():
        return FieldUpdates({})

    if not (pimple_control.firstIter() or dynamic_mesh_controls.moveMeshOuterCorrectors):
        return FieldUpdates({})

    mesh.updateMesh()
    if not mesh.changing():
        return FieldUpdates({})

    update_gravity_head(gh, ghf, mesh, hRef)
    ext.on_mesh_change()
    if not dynamic_mesh_controls.correctPhi:
        return FieldUpdates({"gh": gh, "ghf": ghf})

    assert Uf is not None  # createUfIfPresent.H gives every dynamic mesh a Uf
    # as in correctPhi.H: rebuild phi from the mapped Uf, project, make it relative,
    # then re-evaluate the interface properties on the corrected flux.
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
    ext: Annotated[BoundExtension, momentum_extension],
) -> FieldUpdates:
    """Density-weighted momentum predictor with surface tension."""
    if isinstance(pimple_control, FrozenFlowControl):
        # frozenFlow yes ⇒ interIsoFoam runs `continue` before UEqn.H, so the
        # equation is never assembled and its divSchemes need not be declared.
        return FieldUpdates({})

    mesh = U.mesh()

    # as in UEqn.H: correctBoundaryVelocity before assembly — it feeds the
    # boundary coefficients of ``div(rhoPhi,U)``.
    ext.correct_boundary_velocity(U)
    UEqn = fvVectorMatrix(
        fvm.ddt(rho, U) + fvm.div(rhoPhi, U) + turbulence.divDevRhoReff(rho, U) + ext.terms(rho, U)
    )
    # as in UEqn.H, the call order is the physics: source before relax(),
    # constrain() after, correct() after the solve.
    UEqn.relax()
    ext.constrain(UEqn)

    if pimple_control.momentumPredictor():
        pyf.solve(
            UEqn
            + fvc.reconstruct(
                (mixture.surfaceTensionForce() - ghf * fvc.snGrad(rho) - fvc.snGrad(p_rgh))
                * mesh.magSf()
            )
        )
        ext.correct(U)

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
    pressure_reference: Annotated[PressureReference, "models"],
    mixture: Annotated[MixtureProtocol, "models"],
    last_rAU: Annotated[list[Optional[volScalarField]], "models"],
    ext: Annotated[BoundExtension, pressure_extension],
    Uf: Annotated[Optional[surfaceVectorField], "models"] = None,
    # Default None so a frozen-flow step, where ``momentum`` produced no UEqn, binds.
    UEqn: Optional[fvVectorMatrix] = None,
) -> FieldUpdates:
    """Pressure-velocity coupling for VoF with surface tension and gravity.

    ``Uf`` is the moving-mesh face velocity — ``None`` on a static mesh, where
    every mesh-motion term below reduces to the static form it always had.
    """
    if isinstance(pimple_control, FrozenFlowControl):
        # frozenFlow yes ⇒ interIsoFoam's `continue`: U/p/p_rgh/phi keep their
        # prescribed values and only alpha advects.
        return FieldUpdates({})
    assert UEqn is not None  # non-frozen path always carries the momentum matrix
    pRefCell = pressure_reference.cell
    pRefValue = pressure_reference.value
    needsRef = pressure_reference.needs_ref
    mesh = U.mesh()

    while pimple_control.correct():
        rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
        # Handed to the post-move projection, which re-interpolates it next step.
        last_rAU[0] = rAU
        rAUf = surfaceScalarField(pyf.Word("rAUf"), fvc.interpolate(rAU))

        HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p_rgh))

        # on a moving mesh the correction is built from Uf, not phi — what native's
        # fvc::ddtCorr(U, phi, Uf) dispatches to when mesh.dynamic().
        ddt_corr = fvc.ddtCorr(U, phi) if Uf is None else fvc.ddtCorr(U, Uf)
        # as in pEqn.H: the ddt correction is zeroed inside the MRF cells (it
        # belongs to the absolute frame) before the flux is made relative.
        corr = ext.filter_ddt_corr(fvc.interpolate(rho * rAU) * ddt_corr)
        phiHbyA = surfaceScalarField(
            pyf.Word("phiHbyA"),
            fvc.flux(HbyA) + corr,
        )
        ext.make_relative(phiHbyA)

        # Surface tension + gravity contribution on faces
        phig = surfaceScalarField(
            pyf.Word("phig"),
            (mixture.surfaceTensionForce() - ghf * fvc.snGrad(rho)) * rAUf * mesh.magSf(),
        )
        phiHbyA.assign(phiHbyA + phig)

        # adjustPhi balances the global flux, which only means anything relative to
        # the mesh motion — hence the bracket. It stays behind the same
        # needReference() guard native uses: the pair is not bit-exact.
        if needsRef:
            fvc.makeRelative(phiHbyA, U)
        pyf.adjustPhi(phiHbyA, U, p_rgh)
        if needsRef:
            fvc.makeAbsolute(phiHbyA, U)

        if not ext.constrain_pressure(p_rgh, U, phiHbyA, rAUf):
            pyf.constrainPressure(p_rgh, U, phiHbyA, rAUf)

        while pimple_control.correctNonOrthogonal():
            pEqn = fvScalarMatrix(fvm.laplacian(rAUf, p_rgh) - fvc.div(phiHbyA))
            # as in pEqn.H: pin the matrix at p_rgh's *current* level. pRefValue is a
            # level for the absolute pressure p and would shift the whole solution.
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
        # as in pEqn.H: a second fvOptions.correct(U) closes the corrector,
        # whose reconstruction has just overwritten the predictor's correction.
        ext.correct(U)

        # On a closed domain this also pins p's level to pRefValue and re-levels
        # p_rgh, as interFoam's pEqn.H does.
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
            # as in pEqn.H, after the continuity report: the pressure update in
            # between reads neither phi nor Uf, so the order is preserved.
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
