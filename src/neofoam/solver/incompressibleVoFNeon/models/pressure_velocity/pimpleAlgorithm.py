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
buoyant + capillary source) and hands ``UEqn`` / ``fSigma`` / ``grad_u`` to
``continuity`` through ``FieldUpdates`` (never write-flagged); ``continuity``
runs the ``nCorrectors`` PISO loop and recomputes the static pressure. There is
no outer PIMPLE residual loop — one alpha + momentum + pressure pass per step,
exactly as the legacy ``neoInterFoam`` run body.

This model owns fields: ``U``, ``p_rgh``, ``p``, ``gh``, ``ghf``, ``nut0`` and
models: ``grad_op``, ``pimple_state``, ``pressure_reference``.
"""

from typing import Annotated, Any, Callable

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

from ..incompressibleVoFNeonModel import Model
from ..shared import read_int, read_switch

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


@pimpleNeoN.operation(operation_number="2.1")
def momentum(
    U: Any,
    phi: Any,
    rho: Any,
    mu: Any,
    rhoPhi: Any,
    p_rgh: Any,
    gh: Any,
    ghf: Any,
    nut0: Any,
    alpha1: Any,
    phase: Annotated[dict[str, float], "models"],
    grad_op: Annotated[Any, "models"],
    surf_interp: Annotated[Any, "models"],
    pimple_state: Annotated[Any, "models"],
    neon_runtime: Annotated[Any, "models"],
) -> FieldUpdates:
    """Density-weighted momentum predictor with gravity + surface tension.

    ``UEqn = ddt(rho,U) + div(rhoPhi,U) - laplacian(muf,U) + dev2 laminar
    stress``, solved (when ``momentumPredictor`` is on) against the buoyant +
    capillary source ``reconstruct((fSigma - ghf*snGrad(rho) - snGrad(p_rgh))
    * magSf)``, where ``fSigma = interpolate(sigma*K)*snGrad(alpha1)`` is the
    surface-tension face force. Hands ``UEqn`` / ``fSigma`` / ``grad_u`` to
    ``continuity``.
    """
    rt = neon_runtime
    nn.rotate_old_times(U)
    nn.rotate_old_times(phi)

    muf = surf_interp.interpolate(mu)
    muf.name = "muf"

    # grad(U) for the explicit dev2 laminar viscous stress (interFoam
    # divDevRhoReff). viscous_stress returns -div((mu+nut0)*dev2(T(grad(U))));
    # nut0=0 -> pure laminar. Kept alive (handed to continuity via FieldUpdates)
    # because the operator holds a reference.
    grad_u = grad_op.grad_tensor(U)
    stress_terms = (
        nn.imp.ddt(rho, U)
        + nn.imp.div(rhoPhi, U)
        - nn.imp.laplacian(muf, U)
        + nfb.viscous_stress(mu, nut0, grad_u)
    )
    UEqn = nfb.PDESolverVec3(stress_terms, U, rt)

    sn_rho = nfb.sn_grad(rho)
    sn_prgh = nfb.sn_grad(p_rgh)
    magSf = nfb.mag_sf(rt)
    # Surface-tension face force fSigma = interpolate(sigma*K)*snGrad(alpha1),
    # computed once (curvature depends on alpha1, fixed across the corrector loop).
    fSigma = nfb.surface_tension_force(rt, alpha1, phase["sigma"])
    # Buoyant + capillary face force flux.
    face_force = (fSigma + (-1.0 * ghf) * sn_rho - sn_prgh) * magSf
    src_field = nn.reconstruct(face_force)

    if pimple_state.momentum_predictor:
        UEqn.solve_with_source(nn.exp.source(src_field))
    else:
        UEqn.assemble_and_relax()

    return FieldUpdates({"UEqn": UEqn, "fSigma": fSigma, "grad_u": grad_u, "U": U})


@pimpleNeoN.operation(operation_number="2.2", depends_on=["momentum"])
def continuity(
    U: Any,
    p_rgh: Any,
    p: Any,
    phi: Any,
    rho: Any,
    gh: Any,
    ghf: Any,
    UEqn: Any,
    fSigma: Any,
    pimple_state: Annotated[Any, "models"],
    surf_interp: Annotated[Any, "models"],
    pressure_reference: Annotated[dict[str, Any], "models"],
    neon_runtime: Annotated[Any, "models"],
) -> FieldUpdates:
    """Buoyant ``p_rgh`` PISO pressure correction; recompute static pressure.

    Verbatim port of the corrector loop in ``neoInterFoam.momentum_pressure``:
    ``phig = (fSigma - ghf*snGrad(rho))*rAUf*magSf``, ``phiHbyA = flux(HbyA) +
    rhorAUf*ddtCorr + phig`` (the ddtCorr weight is ``interpolate(rho*rAU)`` —
    the density weight matters across the ~1000x interface jump), then the
    ``laplacian(rAUf, p_rgh) == div(phiHbyA)`` solve and the flux / velocity
    update. ``p = p_rgh + rho*gh`` is the derived static-pressure output.
    """
    state = pimple_state
    rt = neon_runtime
    ddt_scheme = UEqn.ddt_scheme()

    sn_rho = nfb.sn_grad(rho)
    magSf = nfb.mag_sf(rt)
    p_ref_cell = pressure_reference["pRefCell"]
    p_ref_value = pressure_reference["pRefValue"]
    needs_ref = pressure_reference["needsRef"]

    for _ in range(state.n_correctors):
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
        phig = (fSigma + (-1.0 * ghf) * sn_rho) * rAUf * magSf
        phiHbyA = (
            nfb.flux(hByA)
            + rhorAUf * nfb.ddt_flux_corr(U, phi, rt.dt, ddt_scheme)
            + phig
        )

        # Faithful wall fixedFluxPressure (pEqn.H): set the wall p_rgh gradient so
        # the projection cancels the buoyancy/capillary face flux in phiHbyA.
        if _HAS_CONSTRAIN_PRESSURE:
            nfb.constrain_pressure(p_rgh, U, phiHbyA, rAUf)

        pEqn = None
        for _ in range(state.n_non_orth + 1):
            pEqn = nfb.PDESolverScalar(
                nn.imp.laplacian(rAUf, p_rgh) - nn.exp.div(phiHbyA),
                p_rgh,
                rt,
            )
            if needs_ref:
                pEqn.set_reference(p_ref_cell, p_ref_value)
            pEqn.solve()
            p_rgh.correct_boundary_conditions()
        nfb.update_face_velocity(phiHbyA, pEqn, phi)

        sum_local, global_err = nfb.compute_continuity_error(phi, rt)
        state.cumulative_cont_err += global_err
        print(
            f"time step continuity errors : sum local = {sum_local}, "
            f"global = {global_err}, cumulative = {state.cumulative_cont_err}"
        )

        # interFoam velocity correction: U = HbyA + rAU*reconstruct((phig - pEqn.flux())/rAUf).
        # After update_face_velocity, phi = phiHbyA - pEqn.flux(), so pEqn.flux() = phiHbyA - phi
        # and the reconstruct numerator is phig - pEqn.flux() = phig - phiHbyA + phi. Carrying the
        # buoyancy + surface-tension flux this way (not the plain -grad(p_rgh) form) is what keeps
        # the interface velocity physical.
        if _HAS_UPDATE_VELOCITY_BUOYANT:
            numerator = phig - phiHbyA + phi
            nfb.update_velocity_buoyant(hByA, rAU, numerator, rAUf, U)
        else:
            nfb.update_velocity(hByA, rAU, p_rgh, U)
        U.correct_boundary_conditions()

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
