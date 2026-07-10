# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency_resolver
# matches ``param.annotation is Context`` / ``Annotated[..., "models"]`` live.

"""PIMPLE (interFoam) pressure-velocity coupling backed by NeoN.

NeoN implementation of interFoam's ``UEqn.H`` + ``pEqn.H``: a
density-weighted momentum predictor with gravity + surface tension and a
buoyant ``p_rgh`` PISO pressure correction. Laminar — the viscous stress is the
explicit dev2 laminar term (``nfb.viscous_stress(mu, nut0=0, grad(U))``,
matching interFoam's ``divDevRhoReff``), with a zero ``nut0``.

Split into two operations that mirror ``incompressibleFluidNeoN``: ``momentum``
assembles ``UEqn`` (and, when ``momentumPredictor`` is on, solves it against the
buoyant + capillary source) and hands ``UEqn`` / ``grad_u`` to ``continuity``
through ``FieldUpdates`` (never write-flagged); ``continuity`` runs the
``nCorrectors`` PISO loop and recomputes the static pressure. There is no outer
PIMPLE residual loop — one alpha + momentum + pressure pass per step, exactly
as the legacy ``neoInterFoam`` run body.

The gravity + surface-tension face forces are NOT hardcoded here: both
operations fold the ``interfaceForce`` extension point owned by the
``surfaceForces`` model (``..surface_forces``) — ``momentum`` consumes the fold
as ``reconstruct((F - snGrad(p_rgh)) * magSf)``, ``continuity`` as
``phig = F * rAUf * magSf``.

This model owns fields: ``U``, ``p_rgh``, ``p``, ``gh``, ``ghf``, ``nut0`` and
models: ``grad_op``, ``pimple_state``, ``pressure_reference``.
"""

from typing import Annotated, Any, Callable, NamedTuple

import neon._neon as nn  # NeoN Python bindings
from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings

from neofoam.framework.context import Context, FieldUpdates  # noqa: F401
from neofoam.framework.initialization import field, model
from neofoam.framework.operations import (
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata

from ..incompressibleVoFNeoNModel import Model
from ..shared import read_int, read_switch
from ..surface_forces import interfaceForce

pimpleNeoN = Model("PimpleVoFNeoN")

# Whether the NeoN bindings expose the faithful wall fixedFluxPressure
# constraint (``constrain_pressure``). It kills the ~100 m/s spurious wall
# velocity of the zeroGradient approximation on noSlip walls; when the binding
# predates it, the adaptive dt (courant constraint) keeps the run stable, as in
# the legacy neoInterFoam run before the constraint was added.
_HAS_CONSTRAIN_PRESSURE = hasattr(nfb, "constrain_pressure")

# Whether the NeoN bindings expose the buoyant velocity reconstruction
# (``update_velocity_buoyant``). interFoam corrects the cell velocity with
# ``U = HbyA + rAU*reconstruct((phig - pEqn.flux())/rAUf)`` — carrying the
# buoyancy + surface-tension flux. The plain ``update_velocity`` (non-buoyant,
# ``U = HbyA - rAU*grad(p_rgh)``) drops that flux and leaves large spurious
# velocities at the interface; fall back to it only on an older binding.
_HAS_UPDATE_VELOCITY_BUOYANT = hasattr(nfb, "update_velocity_buoyant")


class PimpleVoFState:
    """Per-run PIMPLE loop state shared across the VoF solver operations.

    interFoam runs a single outer corrector; the inner PISO corrector counts
    (``nCorrectors`` / ``nNonOrthogonalCorrectors``) and ``momentumPredictor``
    are read once from the ``PIMPLE`` subdict. ``cumulative_cont_err`` tracks the
    running continuity error for reporting.
    """

    def __init__(
        self, n_correctors: int, n_non_orth: int, momentum_predictor: bool
    ) -> None:
        self.n_correctors = n_correctors
        self.n_non_orth = n_non_orth
        self.momentum_predictor = momentum_predictor
        self.cumulative_cont_err: float = 0.0


@pimpleNeoN.build
def build(self: Any) -> list[Any]:
    """Lazy initializers for the NeoN VoF PIMPLE state.

    The ``U`` / ``p_rgh`` reads and the ``gh`` / ``ghf`` / ``p`` / ``nut0``
    factory fields go through the NeoN field factories on the ``_neon_runtime``
    adapter built in ``create_fields``.
    """

    def create_u(context: dict[str, Any]) -> Any:
        return nfb.read_vector_volume_field(context["_neon_runtime"], "U")

    def create_p_rgh(context: dict[str, Any]) -> Any:
        return nfb.read_scalar_volume_field(context["_neon_runtime"], "p_rgh")

    def create_p(context: dict[str, Any]) -> Any:
        # damBreak ships no `p` field; the static pressure p = p_rgh + rho*gh is
        # a derived output. Create it with calculated BCs, filled after the solve.
        return nfb.create_uniform_volume_field(context["_neon_runtime"], "p", 0.0)

    def create_gh(context: dict[str, Any]) -> Any:
        return nfb.create_gh(context["_neon_runtime"])

    def create_ghf(context: dict[str, Any]) -> Any:
        return nfb.create_ghf(context["_neon_runtime"])

    def create_nut0(context: dict[str, Any]) -> Any:
        # Zero nut for the laminar dev2 viscous stress (interFoam's
        # divDevRhoReff is a pure laminar stress).
        return nfb.create_uniform_volume_field(context["_neon_runtime"], "nut0", 0.0)

    def create_grad_op(context: dict[str, Any]) -> Any:
        return nfb.GaussGreenGrad(context["_neon_runtime"])

    def create_pimple_state(context: dict[str, Any]) -> PimpleVoFState:
        rt = context["_neon_runtime"]
        pimple_dict = rt.fv_solution_dict.subDict("PIMPLE")
        return PimpleVoFState(
            n_correctors=read_int(pimple_dict, "nCorrectors", 1),
            n_non_orth=read_int(pimple_dict, "nNonOrthogonalCorrectors", 0),
            momentum_predictor=read_switch(pimple_dict, "momentumPredictor", True),
        )

    def create_pressure_reference(context: dict[str, Any]) -> dict[str, Any]:
        rt = context["_neon_runtime"]
        cell, value, needs_ref = nfb.set_ref_cell(rt, "p_rgh", "PIMPLE")
        return {"pRefCell": cell, "pRefValue": value, "needsRef": needs_ref}

    return [
        field("U", create_u, depends_on=["_neon_runtime"], write=True),
        field("p_rgh", create_p_rgh, depends_on=["_neon_runtime"], write=True),
        field("gh", create_gh, depends_on=["_neon_runtime"]),
        field("ghf", create_ghf, depends_on=["_neon_runtime"]),
        field("p", create_p, depends_on=["_neon_runtime"]),
        field("nut0", create_nut0, depends_on=["_neon_runtime"]),
        model("grad_op", create_grad_op, depends_on=["_neon_runtime"]),
        model("pimple_state", create_pimple_state, depends_on=["_neon_runtime"]),
        model(
            "pressure_reference",
            create_pressure_reference,
            depends_on=["_neon_runtime", "fields.p_rgh"],
        ),
    ]


def _folded_interface_force(interface_force: Any, ctx: Context) -> Any:
    """Fold the bound ``interfaceForce`` against the live Context.

    The two default contributions (surface tension, gravity) are always wired
    by ``create_fields`` — an empty fold (``None``) is a wiring error.
    """
    face_force = interface_force(ctx)
    if face_force is None:
        raise RuntimeError(
            "surfaceForces: no interface-force contributions are bound "
            "(surface tension + gravity are wired by create_fields)."
        )
    return face_force


class _MomentumAssembly(NamedTuple):
    """``UEqn`` plus the objects its operators hold raw references to.

    The implicit operators and ``viscous_stress`` do NOT copy their operands:
    ``muf`` / ``grad_u`` / ``stress_terms`` must stay alive until ``UEqn`` is
    assembled (solved), so the caller keeps this tuple in scope through the
    solve — and hands ``grad_u`` on to ``continuity`` via ``FieldUpdates``.
    """

    UEqn: Any
    grad_u: Any
    muf: Any
    stress_terms: Any


def _assemble_ueqn(
    U: Any,
    rho: Any,
    mu: Any,
    rhoPhi: Any,
    nut0: Any,
    grad_op: Any,
    surf_interp: Any,
    rt: Any,
) -> _MomentumAssembly:
    """Assemble ``UEqn = ddt(rho,U) + div(rhoPhi,U) - laplacian(muf,U) + dev2``.

    grad(U) feeds the explicit dev2 laminar viscous stress (interFoam
    divDevRhoReff) — viscous_stress returns
    ``-div((mu+nut0)*dev2(T(grad(U))))``, nut0=0 -> pure laminar.
    """
    muf = surf_interp.interpolate(mu)
    muf.name = "muf"
    grad_u = grad_op.grad_tensor(U)
    stress_terms = (
        nn.imp.ddt(rho, U)
        + nn.imp.div(rhoPhi, U)
        - nn.imp.laplacian(muf, U)
        + nfb.viscous_stress(mu, nut0, grad_u)
    )
    return _MomentumAssembly(
        UEqn=nfb.PDESolverVec3(stress_terms, U, rt),
        grad_u=grad_u,
        muf=muf,
        stress_terms=stress_terms,
    )


@pimpleNeoN.operation(operation_number="2.1")
def momentum(
    ctx: Context,
    U: Any,
    phi: Any,
    rho: Any,
    mu: Any,
    rhoPhi: Any,
    p_rgh: Any,
    nut0: Any,
    grad_op: Annotated[Any, "models"],
    surf_interp: Annotated[Any, "models"],
    pimple_state: Annotated[Any, "models"],
    neon_runtime: Annotated[Any, "models"],
    interface_force: interfaceForce,  # type: ignore[valid-type]
) -> FieldUpdates:
    """Density-weighted momentum predictor with the folded interface forces.

    ``UEqn = ddt(rho,U) + div(rhoPhi,U) - laplacian(muf,U) + dev2 laminar
    stress``, solved (when ``momentumPredictor`` is on) against the source
    ``reconstruct((F - snGrad(p_rgh)) * magSf)`` where ``F`` is the folded
    ``interfaceForce`` (surface tension + gravity by default). Hands ``UEqn``
    / ``grad_u`` to ``continuity``.
    """
    rt = neon_runtime
    nn.rotate_old_times(U)
    nn.rotate_old_times(phi)

    # `assembly` keeps muf / grad_u / stress_terms alive through the solve —
    # the UEqn operators hold raw references to them.
    assembly = _assemble_ueqn(U, rho, mu, rhoPhi, nut0, grad_op, surf_interp, rt)
    UEqn = assembly.UEqn
    # interFoam runs a single outer corrector, so the whole step is the final
    # outer pass (OpenFOAM's mesh-data finalIteration flag): the momentum solve
    # selects the "UFinal" solver subdict when the case provides one.
    UEqn.set_final_iter(True)

    # Buoyant + capillary face force flux from the folded contributions.
    face_force = _folded_interface_force(interface_force, ctx)
    src_field = nn.reconstruct((face_force - nfb.sn_grad(p_rgh)) * nfb.mag_sf(rt))

    if pimple_state.momentum_predictor:
        UEqn.solve_with_source(nn.exp.source(src_field))
    else:
        UEqn.assemble_and_relax()

    return FieldUpdates({"UEqn": UEqn, "grad_u": assembly.grad_u, "U": U})


class _FluxPrediction(NamedTuple):
    """One PISO corrector's flux prediction (the pEqn.H setup phase)."""

    rAU: Any
    hByA: Any
    rAUf: Any
    phig: Any
    phiHbyA: Any


def _predict_face_flux(
    UEqn: Any,
    U: Any,
    phi: Any,
    p_rgh: Any,
    rho: Any,
    face_force: Any,
    magSf: Any,
    ddt_scheme: Any,
    surf_interp: Any,
    rt: Any,
) -> _FluxPrediction:
    """rAU/HbyA and the buoyant face flux ``phiHbyA = flux(HbyA) + ddtCorr + phig``."""
    rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
    nfb.constrain_hbya(U, p_rgh, hByA)
    rAUf = surf_interp.interpolate(rAU)
    rAUf.name = "rAUf"
    # ddtCorr weight is interpolate(rho*rAU) (not interpolate(rAU)): near the
    # interface rho jumps ~1000x, so the transient Rhie-Chow correction must
    # carry the density weight. The laplacian / phig terms keep plain rAUf.
    rho_rau = nfb.mul_scalar_volume(rho, rAU)
    rhorAUf = surf_interp.interpolate(rho_rau)
    rhorAUf.name = "rhorAUf"
    phig = face_force * rAUf * magSf
    phiHbyA = (
        nfb.flux(hByA) + rhorAUf * nfb.ddt_flux_corr(U, phi, rt.dt, ddt_scheme) + phig
    )

    # Faithful wall fixedFluxPressure (pEqn.H): set the wall p_rgh gradient so
    # the projection cancels the buoyancy/capillary face flux in phiHbyA.
    if _HAS_CONSTRAIN_PRESSURE:
        nfb.constrain_pressure(p_rgh, U, phiHbyA, rAUf)

    return _FluxPrediction(rAU=rAU, hByA=hByA, rAUf=rAUf, phig=phig, phiHbyA=phiHbyA)


def _solve_pressure(
    p_rgh: Any,
    phi: Any,
    prediction: _FluxPrediction,
    state: Any,
    pressure_reference: dict[str, Any],
    rt: Any,
    final_corrector: bool,
) -> None:
    """Solve ``laplacian(rAUf,p_rgh) == div(phiHbyA)`` (non-orth loop); update phi."""
    p_ref_cell = pressure_reference["pRefCell"]
    p_ref_value = pressure_reference["pRefValue"]
    needs_ref = pressure_reference["needsRef"]

    pEqn: Any = None
    for non_orth in range(state.n_non_orth + 1):
        pEqn = nfb.PDESolverScalar(
            nn.imp.laplacian(prediction.rAUf, p_rgh) - nn.exp.div(prediction.phiHbyA),
            p_rgh,
            rt,
        )
        # interFoam pEqn.H: p_rghEqn.solve(mesh.solver(p_rgh.select(
        # pimple.finalInnerIter()))) — the "p_rghFinal" subdict (relTol 0)
        # applies only to the last corrector's last non-orthogonal solve.
        pEqn.set_final_iter(final_corrector and non_orth == state.n_non_orth)
        if needs_ref:
            pEqn.set_reference(p_ref_cell, p_ref_value)
        pEqn.solve()
        p_rgh.correct_boundary_conditions()
    nfb.update_face_velocity(prediction.phiHbyA, pEqn, phi)


def _report_continuity_error(phi: Any, state: Any, rt: Any) -> None:
    """continuityErrs.H: accumulate and print the time-step continuity errors."""
    sum_local, global_err = nfb.compute_continuity_error(phi, rt)
    state.cumulative_cont_err += global_err
    print(
        f"time step continuity errors : sum local = {sum_local}, "
        f"global = {global_err}, cumulative = {state.cumulative_cont_err}"
    )


def _correct_velocity(
    U: Any, phi: Any, p_rgh: Any, prediction: _FluxPrediction
) -> None:
    """interFoam velocity correction ``U = HbyA + rAU*reconstruct((phig - pEqn.flux())/rAUf)``.

    After update_face_velocity, phi = phiHbyA - pEqn.flux(), so pEqn.flux() =
    phiHbyA - phi and the reconstruct numerator is phig - pEqn.flux() =
    phig - phiHbyA + phi. Carrying the buoyancy + surface-tension flux this way
    (not the plain -grad(p_rgh) form) is what keeps the interface velocity
    physical.
    """
    if _HAS_UPDATE_VELOCITY_BUOYANT:
        numerator = prediction.phig - prediction.phiHbyA + phi
        nfb.update_velocity_buoyant(
            prediction.hByA, prediction.rAU, numerator, prediction.rAUf, U
        )
    else:
        nfb.update_velocity(prediction.hByA, prediction.rAU, p_rgh, U)
    U.correct_boundary_conditions()


@pimpleNeoN.operation(operation_number="2.2", depends_on=["momentum"])
def continuity(
    ctx: Context,
    U: Any,
    p_rgh: Any,
    p: Any,
    phi: Any,
    rho: Any,
    gh: Any,
    UEqn: Any,
    pimple_state: Annotated[Any, "models"],
    surf_interp: Annotated[Any, "models"],
    pressure_reference: Annotated[dict[str, Any], "models"],
    neon_runtime: Annotated[Any, "models"],
    interface_force: interfaceForce,  # type: ignore[valid-type]
) -> FieldUpdates:
    """Buoyant ``p_rgh`` PISO pressure correction; recompute static pressure.

    Port of the corrector loop in ``neoInterFoam.momentum_pressure``, one
    helper per pEqn.H phase: predict the face flux (``phig = F*rAUf*magSf``
    with ``F`` the folded ``interfaceForce``, ``phiHbyA = flux(HbyA) +
    rhorAUf*ddtCorr + phig``), solve ``laplacian(rAUf, p_rgh) ==
    div(phiHbyA)``, report the continuity error, and apply the buoyant
    velocity correction. ``p = p_rgh + rho*gh`` is the derived static-pressure
    output.
    """
    state = pimple_state
    rt = neon_runtime
    ddt_scheme = UEqn.ddt_scheme()
    magSf = nfb.mag_sf(rt)

    # Folded ONCE before the PISO loop (the curvature in the surface-tension
    # contribution depends on alpha1, which is fixed across the correctors).
    face_force = _folded_interface_force(interface_force, ctx)

    for corrector in range(state.n_correctors):
        prediction = _predict_face_flux(
            UEqn, U, phi, p_rgh, rho, face_force, magSf, ddt_scheme, surf_interp, rt
        )
        _solve_pressure(
            p_rgh,
            phi,
            prediction,
            state,
            pressure_reference,
            rt,
            final_corrector=corrector == state.n_correctors - 1,
        )
        _report_continuity_error(phi, state, rt)
        _correct_velocity(U, phi, p_rgh, prediction)

    # Static pressure p = p_rgh + rho*gh (derived output).
    nfb.update_static_pressure(p, p_rgh, rho, gh)

    return FieldUpdates({"U": U, "p_rgh": p_rgh, "p": p, "phi": phi})


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
    """Expose the ``momentum`` and ``continuity`` operations (single pass).

    No inner PIMPLE loop — interFoam runs one outer corrector per step; the
    execution graph steps ``momentum`` then ``continuity`` directly.
    """
    model_ops = Operations()

    wrapped_momentum = pimpleNeoN.wrap_operation(momentum, self)
    wrapped_continuity = pimpleNeoN.wrap_operation(continuity, self)

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
    return model_ops
