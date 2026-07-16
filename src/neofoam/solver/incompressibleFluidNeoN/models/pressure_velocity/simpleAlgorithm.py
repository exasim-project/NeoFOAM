# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency_resolver
# matches ``param.annotation is Context`` / ``Annotated[..., "models"]`` live.

"""SIMPLE (steady state) pressure-velocity coupling backed by NeoN.

Port of ``simpleFoam``'s loop body (``UEqn.H`` / ``pEqn.H``) mirroring
the pybFoam ``incompressibleFluid`` SIMPLE spec: one solver "time step"
is one SIMPLE outer iteration. The momentum equation carries no ddt
operator (the case's ``ddtSchemes`` is ``steadyState``; the C++
turbulence transports read the same scheme and no-op their ddt via
``DdtScheme::Steady``), equations are under-relaxed through the PDE
solver's ``relaxationFactors.equations`` lookup, and the pressure field
is explicitly relaxed after the corrector (``relaxationFactors.fields``),
exactly as in simpleFoam.

Differences from the pybFoam SIMPLE spec, both no-ops for the supported
case class (an open domain whose outlet fixes the pressure level):

* no ``adjustPhi`` — the NeoN stack has no equivalent; a closed domain
  (all-Neumann pressure) would need it for a consistent pressure RHS;
* no ``constrainPressure`` — relevant only for ``fixedFluxPressure``-type
  BCs, which the NeoN pressure field does not carry here.

SIMPLEC (``consistent yes``) is not ported; detection raises so the
case author is not silently served plain SIMPLE.
"""

from typing import Annotated, Any, Callable

import neon._neon as nn  # NeoN Python bindings
from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings

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
from neofoam.solver.pisoControl import PisoControl

from ..incompressibleFluidNeoNModel import Model


simpleNeoN = Model("SimpleNeoN")

# Per-spec fvSchemes / fvSolution slices — schema-only (see pimpleAlgorithm:
# at run time the NeoN runtime maps the on-disk dicts instead).
SimpleNeoNFvSchemes = simpleNeoN.config(fvSchemes)
SimpleNeoNFvSolution = simpleNeoN.config(fvSolution)

# Optional SIMPLE control keys read by ``set_ref_cell``.
SimpleNeoNFvSolution.add_controls("SIMPLE", pRefCell=int, pRefValue=float)

# 0/<name> field declarations (same arm sets as the PIMPLE spec; the actual
# NeoN reads are emitted by ``@simpleNeoN.build`` below).
simpleNeoN.field(
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
simpleNeoN.field(
    "p",
    dimensions=[0, 2, -2, 0, 0, 0, 0],
    value_type=Scalar,
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


def _read_int(d: Any, key: str, default: int) -> int:
    return int(d.get_int(key)) if d.contains(key) else default


def _read_switch(d: Any, key: str, default: bool) -> bool:
    """Read an OpenFOAM on/off switch from a converted NeoN dictionary.

    The OpenFOAM→NeoN dict conversion stores switch words (``yes``/``no``)
    as strings; a wrong-typed ``get_*`` is fatal in NeoN (no catchable
    exception), so this must read the string directly.
    """
    if not d.contains(key):
        return default
    return d.get_string(key).strip().lower() in ("yes", "true", "on", "1")


def _reduce_u(stats: Any) -> tuple[float, float]:
    """Max-component reduction: Ux/Uy/Uz entries -> one {init, final} pair."""
    mi = 0.0
    mf = 0.0
    for e in stats.entries:
        mi = max(mi, e.initial_residual)
        mf = max(mf, e.final_residual)
    return (mi, mf)


class SimpleNeoNState:
    """Per-run SIMPLE loop state shared across the solver operations.

    ``piso`` supplies the self-resetting non-orthogonal corrector loop and
    the momentumPredictor flag (SIMPLE has exactly one pressure correction,
    so its PISO-corrector count is never used). ``residuals`` is reported
    per step; ``outer_open`` gates the one-pass inner loop.
    """

    def __init__(self, piso: PisoControl) -> None:
        self.piso = piso
        self.residuals: dict[str, tuple[float, float]] = {}
        self.cumulative_cont_err: float = 0.0
        self.outer_open: bool = True


@simpleNeoN.build
def build(self: Any) -> list[Any]:
    """Lazy initializers for the NeoN SIMPLE state (see pimpleAlgorithm)."""

    def create_p(context: dict[str, Any]) -> Any:
        return nfb.read_scalar_volume_field(context["_neon_runtime"], "p")

    def create_u(context: dict[str, Any]) -> Any:
        return nfb.read_vector_volume_field(context["_neon_runtime"], "U")

    def create_phi(context: dict[str, Any]) -> Any:
        return nfb.create_phi(context["_neon_runtime"], "U")

    def create_simple_state(context: dict[str, Any]) -> SimpleNeoNState:
        rt = context["_neon_runtime"]
        simple_dict = rt.fv_solution_dict.subDict("SIMPLE")
        if _read_switch(simple_dict, "consistent", False):
            raise NotImplementedError(
                "incompressibleFluidNeoN: SIMPLEC (consistent yes) is not ported"
            )
        piso = PisoControl(
            n_correctors=1,
            n_non_orthogonal_correctors=_read_int(
                simple_dict, "nNonOrthogonalCorrectors", 0
            ),
            momentum_predictor=_read_switch(simple_dict, "momentumPredictor", True),
        )
        return SimpleNeoNState(piso)

    def create_pressure_reference(context: dict[str, Any]) -> dict[str, Any]:
        rt = context["_neon_runtime"]
        cell, value, needs_ref = nfb.set_ref_cell(rt, "p", "SIMPLE")
        return {"pRefCell": cell, "pRefValue": value, "needsRef": needs_ref}

    def create_surf_interp(context: dict[str, Any]) -> Any:
        rt = context["_neon_runtime"]
        return nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )

    def create_grad_op(context: dict[str, Any]) -> Any:
        return nfb.GaussGreenGrad(context["_neon_runtime"])

    return [
        field("p", create_p, depends_on=["_neon_runtime"], write=True),
        field("U", create_u, depends_on=["_neon_runtime"], write=True),
        field("phi", create_phi, depends_on=["_neon_runtime", "fields.U"]),
        model("simple_state", create_simple_state, depends_on=["_neon_runtime"]),
        model(
            "pressure_reference",
            create_pressure_reference,
            depends_on=["_neon_runtime", "fields.p"],
        ),
        model("surf_interp", create_surf_interp, depends_on=["_neon_runtime"]),
        model("grad_op", create_grad_op, depends_on=["_neon_runtime"]),
    ]


def inner_loop(ctx: Context) -> bool:
    """One momentum+continuity pass per solver step (mirrors SimpleControl)."""
    state = ctx.models["simple_state"]
    if state.outer_open:
        state.outer_open = False
        return True
    state.outer_open = True
    return False


@simpleNeoN.operation(operation_number="2.0")
def rotate_and_report(
    simple_state: Annotated[Any, "models"],
) -> None:
    """Start-of-step bookkeeping: reset the per-step residual report.

    Steady state — nothing rotates (no ddt operator reads an oldTime) and
    simpleFoam prints no Courant number.
    """
    simple_state.residuals = {}


@simpleNeoN.operation(operation_number="2.1")
@SimpleNeoNFvSchemes.add(
    div=["div(phi,U)", "div((nuEff*dev2(T(grad(U)))))"],
    grad="grad(U)",
    laplacian="laplacian(nuEff,U)",
)
@SimpleNeoNFvSolution.add("U")
def momentum(
    U: Any,
    phi: Any,
    p: Any,
    simple_state: Annotated[Any, "models"],
    turbulence: Annotated[Any, "models"],
    grad_op: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    neon_runtime: Annotated[Any, "models"],
) -> FieldUpdates:
    """Assemble (and optionally solve) the steady momentum equation.

    No ddt operator; the PDE solver applies the ``relaxationFactors.equations``
    under-relaxation for ``U`` during assembly, matching ``UEqn.relax()``.
    """
    state = simple_state

    # Snapshot p for the explicit field under-relaxation after the pressure
    # solve (consumed by continuity).
    prev_p = nfb.field_relaxation_snapshot(p)

    # grad(U) for the explicit dev2 viscous stress; kept alive via FieldUpdates.
    grad_u = grad_op.grad_tensor(U)

    UEqn = nfb.PDESolverVec3(
        nn.imp.div(phi, U)
        - nn.imp.laplacian(turbulence.nu_eff(), U)
        + nfb.viscous_stress(nu_vol, turbulence.nut(), grad_u),
        U,
        neon_runtime,
    )

    if state.piso.momentum_predictor():
        stats_u = UEqn.solve_with_source(-1.0 * nn.exp.grad(p))
        state.residuals["U"] = _reduce_u(stats_u)
    else:
        # Relax unconditionally so computeRAUandHByA reads the relaxed diagonal.
        UEqn.assemble_and_relax()

    return FieldUpdates({"UEqn": UEqn, "prev_p": prev_p, "grad_u": grad_u, "U": U})


@simpleNeoN.operation(operation_number="2.2", depends_on=["momentum"])
@SimpleNeoNFvSchemes.add(
    grad="grad(p)",
    laplacian="laplacian(rAUf,p)",
    interpolation=["flux(HbyA)", "interpolate(rAU)"],
    snGrad="snGrad(p)",
)
@SimpleNeoNFvSolution.add("p")
def continuity(
    U: Any,
    p: Any,
    phi: Any,
    UEqn: Any,
    prev_p: Any,
    simple_state: Annotated[Any, "models"],
    surf_interp: Annotated[Any, "models"],
    pressure_reference: Annotated[dict[str, Any], "models"],
    neon_runtime: Annotated[Any, "models"],
) -> FieldUpdates:
    """The SIMPLE pressure correction: solve, flux + velocity update, p relax.

    Verbatim port of ``simpleFoam``'s ``pEqn.H`` (minus adjustPhi /
    constrainPressure — see the module docstring): plain ``flux(HbyA)``
    with no ddt flux correction, one pressure correction, explicit p
    field relaxation before the velocity update.
    """
    state = simple_state
    rt = neon_runtime
    p_ref_cell = pressure_reference["pRefCell"]
    p_ref_value = pressure_reference["pRefValue"]
    needs_ref = pressure_reference["needsRef"]

    rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
    nfb.constrain_hbya(U, p, hByA)

    rAUf = surf_interp.interpolate(rAU)
    rAUf.name = "rAUf"

    phiHbyA = nfb.flux(hByA)

    have_p_res = False
    while state.piso.correct_non_orthogonal():
        pEqn = nfb.PDESolverScalar(
            nn.imp.laplacian(rAUf, p) - nn.exp.div(phiHbyA),
            p,
            rt,
        )
        if needs_ref:
            pEqn.set_reference(p_ref_cell, p_ref_value)

        stats_p = pEqn.solve()
        if not have_p_res:
            entry = stats_p.entries[0]
            state.residuals["p"] = (entry.initial_residual, entry.final_residual)
            have_p_res = True
        p.correct_boundary_conditions()

        if state.piso.final_non_orthogonal_iter():
            nfb.update_face_velocity(phiHbyA, pEqn, phi)

    sum_local, global_err = nfb.compute_continuity_error(phi, rt)
    state.cumulative_cont_err += global_err
    print(
        f"time step continuity errors : sum local = {sum_local}, "
        f"global = {global_err}, cumulative = {state.cumulative_cont_err}"
    )

    # Explicit pressure under-relaxation (relaxationFactors.fields), then the
    # velocity correction — the simpleFoam ordering.
    nfb.apply_field_relaxation(
        p,
        prev_p,
        nfb.lookup_field_relaxation(rt.fv_solution_dict, p.name, False),
    )
    p.correct_boundary_conditions()

    nfb.update_velocity(hByA, rAU, p, U)
    U.correct_boundary_conditions()

    return FieldUpdates({"U": U, "p": p, "phi": phi})


@simpleNeoN.operation(operation_number="2.3")
def turbulence_correct(
    U: Any,
    phi: Any,
    turbulence: Annotated[Any, "models"],
    neon_runtime: Annotated[Any, "models"],
) -> None:
    """Update the turbulence model once per SIMPLE iteration."""
    turbulence.correct(U, phi, neon_runtime)


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


@simpleNeoN.operation_collection
def collected_operations(self: Any) -> Operations:
    """Expose the same op-name surface as the PIMPLE spec.

    The solver's execution graph consumes ``inner_loop`` / ``rotate_and_report``
    / ``momentum`` / ``continuity`` / ``turbulence_correct`` by name, so the
    SIMPLE spec slots in without solver changes.
    """
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=IterativeOp(inner_loop),
            metadata=OperationMetadata(op_name="inner_loop"),
        )
    )

    wrapped_rotate = wrap_with_dependency_resolution(
        rotate_and_report, self, simpleNeoN._dependency_resolver
    )
    wrapped_momentum = wrap_with_dependency_resolution(
        momentum, self, simpleNeoN._dependency_resolver
    )
    wrapped_continuity = wrap_with_dependency_resolution(
        continuity, self, simpleNeoN._dependency_resolver
    )
    wrapped_turbulence = wrap_with_dependency_resolution(
        turbulence_correct, self, simpleNeoN._dependency_resolver
    )

    model_ops.add(
        _alias_operation(
            wrapped_rotate, operation_name="rotate_and_report", depends_on=[]
        )
    )
    model_ops.add(
        _alias_operation(wrapped_momentum, operation_name="momentum", depends_on=[])
    )
    model_ops.add(
        _alias_operation(
            wrapped_continuity,
            operation_name="continuity",
            depends_on=["momentum"],
        )
    )
    model_ops.add(
        _alias_operation(
            wrapped_turbulence, operation_name="turbulence_correct", depends_on=[]
        )
    )
    return model_ops
