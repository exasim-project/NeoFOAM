# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency_resolver
# matches ``param.annotation is Context`` / ``Annotated[..., "models"]`` live.

"""PIMPLE pressure-velocity coupling backed by NeoN.

Framework port of the loop body of
:mod:`neofoam.solver.neoPimpleFoam` (itself a port of
``examples/neoPimpleFoam/neoPimpleFoam.cpp``): a PISO inner corrector nested
inside an outer, residual-driven PIMPLE loop, assembled through the NeoN
bindings (``neon._neon`` / ``neofoam.neofoam_bindings``).

The momentum stress follows ``pimpleFoam``'s ``divDevReff(U)`` decomposition,
``-laplacian(nuEff,U) - div(nuEff*dev2(T(grad(U))))``: the implicit laplacian
plus the explicit dev2 viscous-stress term. ``nuEff``/``nut`` come from the
runtime-selected turbulence model (created in ``create_fields``), fixed
across the PIMPLE loop and corrected once per time step (the solver's
``turbulence_correct`` op), as in OpenFOAM.

Cross-op state (the outer-loop control, the PISO corrector counters, the
residual map the outer loop converges on, and the running continuity error)
lives on :class:`PimpleNeoNState` (``ctx.models["pimple_state"]``);
``UEqn`` / ``prev_p`` / ``grad_u`` are handed from ``momentum`` to
``continuity`` through ``FieldUpdates`` (never write-flagged).
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


pimpleNeoN = Model("PimpleNeoN")

# Per-spec fvSchemes / fvSolution slices — schema-only (mirrors the pybFoam
# pimple spec so ``configurations(solver)`` surfaces the same case scaffold;
# at run time the NeoN runtime maps the on-disk dicts via ``map_fv_schemes`` /
# ``map_fv_solution`` instead of loading these classes).
PimpleNeoNFvSchemes = pimpleNeoN.config(fvSchemes)
PimpleNeoNFvSolution = pimpleNeoN.config(fvSolution)

# Optional PIMPLE control keys read by ``set_ref_cell``: a closed domain (no
# fixed-pressure BC) needs a pressure reference; an open domain doesn't.
PimpleNeoNFvSolution.add_controls("PIMPLE", pRefCell=int, pRefValue=float)

# 0/<name> field declarations PIMPLE owns — schema-only here: the framework's
# read-field synthesis is pybFoam-backed, so the actual NeoN reads are emitted
# by ``@pimpleNeoN.build`` below (``nfb.read_*_volume_field``) instead of
# ``synthesize_init_step``.
pimpleNeoN.field(
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
pimpleNeoN.field(
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


class PimpleNeoNState:
    """Per-run PIMPLE loop state shared across the solver operations.

    ``control`` is the C++ residual-driven outer loop (``nfb.PimpleControl``),
    ``piso`` the pure-Python PISO corrector counters. ``residuals`` is the map
    the outer loop converges on — reset once per time step, written by the
    momentum/continuity operations, read by ``control.loop`` (nanobind copies
    the dict per call, so the Python dict stays authoritative).
    """

    def __init__(self, control: Any, piso: PisoControl) -> None:
        self.control = control
        self.piso = piso
        self.residuals: dict[str, tuple[float, float]] = {}
        self.cumulative_cont_err: float = 0.0


@pimpleNeoN.build
def build(self: Any) -> list[Any]:
    """Lazy initializers for the NeoN PIMPLE state.

    Unlike the pybFoam pimple spec, the ``U`` / ``p`` reads are emitted here
    (not synthesized from the field declarations): they go through the NeoN
    field factories and need the ``_neon_runtime`` adapter, an init-only
    resource built in ``create_fields``.
    """

    def create_p(context: dict[str, Any]) -> Any:
        return nfb.read_scalar_volume_field(context["_neon_runtime"], "p")

    def create_u(context: dict[str, Any]) -> Any:
        return nfb.read_vector_volume_field(context["_neon_runtime"], "U")

    def create_phi(context: dict[str, Any]) -> Any:
        return nfb.create_phi(context["_neon_runtime"], "U")

    def create_pimple_state(context: dict[str, Any]) -> PimpleNeoNState:
        rt = context["_neon_runtime"]
        # Inner-corrector counts live in the "PIMPLE" subdict for a stock
        # pimpleFoam case (there is no "PISO" block); OpenFOAM's
        # solutionControl defaults momentumPredictor to true.
        pimple_dict = rt.fv_solution_dict.subDict("PIMPLE")
        piso = PisoControl(
            n_correctors=_read_int(pimple_dict, "nCorrectors", 1),
            n_non_orthogonal_correctors=_read_int(
                pimple_dict, "nNonOrthogonalCorrectors", 0
            ),
            momentum_predictor=_read_switch(pimple_dict, "momentumPredictor", True),
        )
        return PimpleNeoNState(nfb.PimpleControl(rt.fv_solution_dict), piso)

    def create_pressure_reference(context: dict[str, Any]) -> dict[str, Any]:
        rt = context["_neon_runtime"]
        cell, value, needs_ref = nfb.set_ref_cell(rt, "p", "PIMPLE")
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
        model("pimple_state", create_pimple_state, depends_on=["_neon_runtime"]),
        model(
            "pressure_reference",
            create_pressure_reference,
            depends_on=["_neon_runtime", "fields.p"],
        ),
        model("surf_interp", create_surf_interp, depends_on=["_neon_runtime"]),
        model("grad_op", create_grad_op, depends_on=["_neon_runtime"]),
    ]


def inner_loop(ctx: Context) -> bool:
    """Outer PIMPLE predicate — ``nfb.PimpleControl.loop`` on the live residuals."""
    state = ctx.models["pimple_state"]
    return bool(state.control.loop(state.residuals))


@pimpleNeoN.operation(operation_number="2.0")
def rotate_and_report(
    U: Any,
    phi: Any,
    turbulence: Annotated[Any, "models"],
    pimple_state: Annotated[Any, "models"],
    neon_runtime: Annotated[Any, "models"],
) -> None:
    """Start-of-step bookkeeping: rotate old times, report the Courant number.

    Maps the top of the legacy time-loop body (``nn.rotate_old_times`` +
    ``compute_co_num`` print). Also resets the residual map the outer PIMPLE
    loop converges on — a fresh dict per time step, as in the legacy port.
    """
    nn.rotate_old_times(U)
    nn.rotate_old_times(phi)
    turbulence.rotate_old_times()

    max_co, mean_co = nn.compute_co_num(phi, neon_runtime.dt)
    print(f"Courant Number mean: {mean_co:.6f} max: {max_co:.6f}")

    pimple_state.residuals = {}


@pimpleNeoN.operation(operation_number="2.1")
@PimpleNeoNFvSchemes.add(
    ddt="ddt(U)",
    # ``div(phi,U)`` (convection) + the viscous-stress divergence emitted by
    # ``divDevReff(U)`` — both required for the momentum predictor.
    div=["div(phi,U)", "div((nuEff*dev2(T(grad(U)))))"],
    grad="grad(U)",
    laplacian="laplacian(nuEff,U)",
    # ``flux(U)`` builds the initial face flux phi (``nfb.create_phi``); the NeoN
    # runtime looks it up in interpolationSchemes, so it must be declared or the
    # case aborts with *Entry 'flux(U)' not found*.
    interpolation="flux(U)",
)
@PimpleNeoNFvSolution.add("U")
def momentum(
    U: Any,
    phi: Any,
    p: Any,
    pimple_state: Annotated[Any, "models"],
    turbulence: Annotated[Any, "models"],
    grad_op: Annotated[Any, "models"],
    nu_vol: Annotated[Any, "models"],
    neon_runtime: Annotated[Any, "models"],
) -> FieldUpdates:
    """Assemble (and optionally solve) the momentum equation — one outer pass.

    ``final_iter`` is queried live from the outer control: its counter was
    advanced by the ``inner_loop`` predicate call that admitted this pass, so
    it matches the legacy top-of-outer-corrector snapshot.
    """
    final_iter = bool(pimple_state.control.final_iter())

    # Snapshot p at the top of each outer corrector to blend against after the
    # pressure solve (explicit field under-relaxation, consumed by continuity).
    prev_p = nfb.field_relaxation_snapshot(p)

    # grad(U) for the explicit dev2 viscous stress. Keep ``grad_u`` alive for
    # the whole outer pass (the operator holds a reference) — it rides to
    # ctx.fields via the FieldUpdates below.
    grad_u = grad_op.grad_tensor(U)

    UEqn = nfb.PDESolverVec3(
        nn.imp.ddt(U)
        + nn.imp.div(phi, U)
        - nn.imp.laplacian(turbulence.nu_eff(), U)
        + nfb.viscous_stress(nu_vol, turbulence.nut(), grad_u),
        U,
        neon_runtime,
    )

    if UEqn.ddt_scheme() not in (nfb.DdtScheme.BDF1, nfb.DdtScheme.BDF2):
        raise RuntimeError(
            "incompressibleFluidNeoN: steadyState ddt unsupported (BDF1/BDF2 only)"
        )

    UEqn.set_final_iter(final_iter)

    if pimple_state.piso.momentum_predictor():
        stats_u = UEqn.solve_with_source(-1.0 * nn.exp.grad(p))
        pimple_state.residuals["U"] = _reduce_u(stats_u)
    else:
        # Relax unconditionally so computeRAUandHByA reads the relaxed
        # diagonal even when the momentum predictor is disabled.
        UEqn.assemble_and_relax()

    return FieldUpdates({"UEqn": UEqn, "prev_p": prev_p, "grad_u": grad_u, "U": U})


@pimpleNeoN.operation(operation_number="2.2", depends_on=["momentum"])
@PimpleNeoNFvSchemes.add(
    grad="grad(p)",
    laplacian="laplacian(rAUf,p)",
    interpolation=["flux(HbyA)", "interpolate(rAU)", "dotInterpolate(S,U_0)"],
    snGrad="snGrad(p)",
)
@PimpleNeoNFvSolution.add("p")
def continuity(
    U: Any,
    p: Any,
    phi: Any,
    UEqn: Any,
    prev_p: Any,
    pimple_state: Annotated[Any, "models"],
    surf_interp: Annotated[Any, "models"],
    pressure_reference: Annotated[dict[str, Any], "models"],
    neon_runtime: Annotated[Any, "models"],
) -> FieldUpdates:
    """The PISO corrector: pressure solve, flux + velocity update.

    Verbatim port of the legacy ``while piso.correct():`` block; ``UEqn`` and
    ``prev_p`` were parked in ``ctx.fields`` by ``momentum`` this outer pass.
    """
    state = pimple_state
    rt = neon_runtime
    final_iter = bool(state.control.final_iter())
    p_ref_cell = pressure_reference["pRefCell"]
    p_ref_value = pressure_reference["pRefValue"]
    needs_ref = pressure_reference["needsRef"]
    ddt_scheme = UEqn.ddt_scheme()

    p_res: tuple[float, float] = (0.0, 0.0)
    have_p_res = False

    while state.piso.correct():
        rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
        nfb.constrain_hbya(U, p, hByA)

        rAUf = surf_interp.interpolate(rAU)
        rAUf.name = "rAUf"

        phiHbyA = nfb.flux(hByA) + rAUf * nfb.ddt_flux_corr(U, phi, rt.dt, ddt_scheme)

        while state.piso.correct_non_orthogonal():
            pEqn = nfb.PDESolverScalar(
                nn.imp.laplacian(rAUf, p) - nn.exp.div(phiHbyA),
                p,
                rt,
            )
            pEqn.set_final_iter(final_iter)

            if needs_ref:
                pEqn.set_reference(p_ref_cell, p_ref_value)

            stats_p = pEqn.solve()
            if not have_p_res:
                entry = stats_p.entries[0]
                p_res = (entry.initial_residual, entry.final_residual)
                have_p_res = True
            p.correct_boundary_conditions()

            if state.piso.final_non_orthogonal_iter():
                nfb.update_face_velocity(phiHbyA, pEqn, phi)

        nfb.apply_field_relaxation(
            p,
            prev_p,
            nfb.lookup_field_relaxation(rt.fv_solution_dict, p.name, final_iter),
        )
        p.correct_boundary_conditions()

        sum_local, global_err = nfb.compute_continuity_error(phi, rt)
        state.cumulative_cont_err += global_err
        print(
            f"time step continuity errors : sum local = {sum_local}, "
            f"global = {global_err}, cumulative = {state.cumulative_cont_err}"
        )

        nfb.update_velocity(hByA, rAU, p, U)
        U.correct_boundary_conditions()

    if have_p_res:
        state.residuals["p"] = p_res

    return FieldUpdates({"U": U, "p": p, "phi": phi})


@pimpleNeoN.operation(operation_number="2.3")
def turbulence_correct(
    U: Any,
    phi: Any,
    turbulence: Annotated[Any, "models"],
    neon_runtime: Annotated[Any, "models"],
) -> None:
    """Update the turbulence model once per time step (after the PIMPLE loop).

    Solves the nuTilda transport PDE and refreshes nut/gradU for SA-DDES; a
    no-op recompute for laminar.
    """
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


@pimpleNeoN.operation_collection
def collected_operations(self: Any) -> Operations:
    """Wrap momentum/continuity inside the PIMPLE inner loop.

    Also exposes the once-per-step ``rotate_and_report`` (before the inner
    loop) and ``turbulence_correct`` (after it) the solver's execution graph
    steps explicitly.
    """
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=IterativeOp(inner_loop),
            metadata=OperationMetadata(op_name="inner_loop"),
        )
    )

    wrapped_rotate = wrap_with_dependency_resolution(
        rotate_and_report, self, pimpleNeoN._dependency_resolver
    )
    wrapped_momentum = wrap_with_dependency_resolution(
        momentum, self, pimpleNeoN._dependency_resolver
    )
    wrapped_continuity = wrap_with_dependency_resolution(
        continuity, self, pimpleNeoN._dependency_resolver
    )
    wrapped_turbulence = wrap_with_dependency_resolution(
        turbulence_correct, self, pimpleNeoN._dependency_resolver
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
