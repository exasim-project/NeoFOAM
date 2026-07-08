# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency_resolver
# matches ``param.annotation is Context`` / ``Annotated[..., "models"]`` live.

"""AlphaAdvection model for incompressibleVoFNeon — NeoN MULES phase advection.

NeoN implementation of interFoam's ``alphaEqn.H``: one MULES-limited
phase-fraction step per time step, backed by the NeoN-core FCT
limiter (``nfb.mules_explicit_solve``). The high-order face flux
``alphaPhi = linearInterp(alpha1) * phi`` is limited in place (no clamp —
boundedness comes from the limiter) while ``alpha1`` is advanced conservatively;
``rho`` / ``mu`` are rebuilt from the limited flux and ``rhoPhi`` is recomputed
so the momentum ``ddt(rho,U)`` and continuity stay conservation-consistent.

Fields provided (owned by this model):
  - phi     (face volumetric flux, from U)
  - alpha1  (phase-1 volume fraction, ``alpha.water``)
  - rho     (mixture density)
  - mu      (mixture dynamic viscosity)
  - rhoPhi  (density-weighted face flux)

Models provided:
  - phase          (two-phase transport property dict)
  - surf_interp    (linear SurfaceInterpolationScalar, shared with pimple)
  - alpha_settings (MULES limiter sweeps ``nLimiterIter``)

Operations:
  - alpha_advection  (one MULES step + rho/mu/rhoPhi rebuild)
"""

from typing import Annotated, Any

import neon._neon as nn  # NeoN Python bindings
from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings

from neofoam.framework.context import FieldUpdates
from neofoam.framework.dependency_resolver import wrap_with_dependency_resolution
from neofoam.framework.initialization import field, model
from neofoam.framework.operations import (
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata

from ..incompressibleVoFNeonModel import Model

alpha_advection_model = Model("AlphaAdvectionNeoN")


def _read_int(d: Any, key: str, default: int) -> int:
    return int(d.get_int(key)) if d.contains(key) else default


def _read_float(d: Any, key: str, default: float) -> float:
    """Read a scalar, tolerating int storage (``cAlpha 1`` parses as an int, so
    ``get_double`` would raise ``bad any_cast``)."""
    if not d.contains(key):
        return default
    try:
        return float(d.get_double(key))
    except Exception:
        return float(d.get_int(key))


def _alpha_solver_dict(rt: Any) -> Any:
    """The ``alpha.water`` alpha-controls subdict, or ``None``.

    The user's subdict is keyed by an OpenFOAM regex (damBreak: ``"alpha.water.*"``)
    and carries the MULES controls (``MULESCorr`` / ``nAlphaCorr`` / ``cAlpha`` /
    ``nLimiterIter``); the NeoN dictionary stores the pattern verbatim and does no
    regex matching, so look it up by its ``alpha.water`` prefix. ``create_fields``
    also registers a MULES-key-*stripped* copy under the exact key ``"alpha.water"``
    for the predictor's linear solver, so PREFER a prefixed key that still carries
    the controls (any ``alpha.water`` key other than the bare ``"alpha.water"``),
    falling back to the exact key only if it is the sole match.
    """
    solvers = rt.fv_solution_dict.subDict("solvers")
    keys = [k for k in solvers.keys() if k.startswith("alpha.water")]
    if not keys:
        return None
    key = next((k for k in keys if k != "alpha.water"), keys[0])
    return solvers.subDict(key)


def _read_switch(d: Any, key: str, default: bool) -> bool:
    """Read an OpenFOAM on/off switch, tolerating word or bool storage."""
    if d is None or not d.contains(key):
        return default
    try:
        return bool(d.get_bool(key))
    except Exception:
        return d.get_string(key).strip().lower() in ("yes", "true", "on", "1")


def _read_alpha_controls(rt: Any) -> dict[str, Any]:
    """MULES sweeps + interface-compression coefficient (alphaEqn.H controls).

    ``nLimiterIter`` = FCT limiter sweeps (default 3); ``cAlpha`` = interface
    compression coefficient (default 1, interFoam's damBreak value) — 0 disables
    compression. ``MULESCorr`` selects the implicit-predictor + corrector path
    (interFoam's semi-implicit MULES); ``nAlphaCorr`` = number of correctors used
    only on the MULESCorr path (the explicit path always advances once).
    """
    d = _alpha_solver_dict(rt)
    if d is None:
        return {
            "n_limiter_iter": 3,
            "c_alpha": 1.0,
            "mules_corr": False,
            "n_alpha_corr": 1,
        }
    return {
        "n_limiter_iter": _read_int(d, "nLimiterIter", 3),
        "c_alpha": _read_float(d, "cAlpha", 1.0),
        "mules_corr": _read_switch(d, "MULESCorr", False),
        "n_alpha_corr": _read_int(d, "nAlphaCorr", 1),
    }


# ---------------------------------------------------------------------------
# Build: register field/model initialisation steps provided by this model
# ---------------------------------------------------------------------------


@alpha_advection_model.build
def build(self: Any) -> list[Any]:
    """Lazy initializers for the NeoN alpha-advection state.

    Emits the NeoN field reads / factory fields directly (not synthesized from
    field declarations, which is a pybFoam-only path): they go through the NeoN
    field factories on the ``_neon_runtime`` adapter built in ``create_fields``.
    """

    def create_phase(context: dict[str, Any]) -> Any:
        return nfb.read_two_phase_transport_properties(context["_neon_runtime"])

    def create_alpha1(context: dict[str, Any]) -> Any:
        return nfb.read_scalar_volume_field(context["_neon_runtime"], "alpha.water")

    def create_phi(context: dict[str, Any]) -> Any:
        return nfb.create_phi(context["_neon_runtime"], "U")

    def create_rho_phi(context: dict[str, Any]) -> Any:
        return nfb.create_rho_phi(context["_neon_runtime"])

    def create_rho(context: dict[str, Any]) -> Any:
        return nfb.create_mixture_density(context["_neon_runtime"])

    def create_mu(context: dict[str, Any]) -> Any:
        return nfb.create_mixture_viscosity(context["_neon_runtime"])

    def create_surf_interp(context: dict[str, Any]) -> Any:
        rt = context["_neon_runtime"]
        return nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )

    def create_alpha_settings(context: dict[str, Any]) -> dict[str, Any]:
        return _read_alpha_controls(context["_neon_runtime"])

    return [
        model("phase", create_phase, depends_on=["_neon_runtime"]),
        field(
            "alpha1", create_alpha1, depends_on=["_neon_runtime"], write=True
        ),
        field("phi", create_phi, depends_on=["_neon_runtime", "fields.U"]),
        field(
            "rhoPhi", create_rho_phi, depends_on=["_neon_runtime", "fields.phi"]
        ),
        field(
            "rho",
            create_rho,
            depends_on=["_neon_runtime", "fields.alpha1"],
        ),
        field(
            "mu",
            create_mu,
            depends_on=["_neon_runtime", "fields.alpha1"],
        ),
        model("surf_interp", create_surf_interp, depends_on=["_neon_runtime"]),
        model(
            "alpha_settings", create_alpha_settings, depends_on=["_neon_runtime"]
        ),
    ]


def _mules_corr_solve(
    rt: Any, alpha1: Any, phi: Any, c_alpha: float, n_limiter: int, n_alpha_corr: int
) -> Any:
    """interFoam semi-implicit MULES (MULESCorr) — predictor + corrector loop.

    Port of the ``MULESCorr`` branch of ``alphaEqn.H``. The predictor solves the
    implicit upwind transport ``ddt(alpha1) + div_upwind(phi, alpha1) == 0`` (via
    the generic scalar PDE solver — the ``div(phi,alpha.water)`` upwind scheme and
    the exact ``alpha.water`` Ginkgo solver dict are registered in
    ``create_fields``), giving a bounded ``alpha1`` whose upwind flux is the FCT
    donor base ``alphaPhi10``. Each of the ``nAlphaCorr`` correctors rebuilds the
    high-order flux ``alphaPhiUn`` (vanLeer + ``cAlpha`` compression), FCT-limits
    the antidiffusive correction ``alphaPhiUn - alphaPhi10`` with ``mules_correct``
    (under-relaxed for correctors past the first, ``relaxOld=0.5``), and
    accumulates the limited correction into ``alphaPhi10``. Returns ``alphaPhi10``
    (the conservative density-weighted flux the ``rhoPhi`` rebuild consumes).
    """
    # Implicit upwind predictor (alpha1.oldTime() already rotated by the caller).
    pred = nfb.PDESolverScalar(nn.imp.ddt(alpha1) + nn.imp.div(phi, alpha1), alpha1, rt)
    pred.solve()
    alpha1.correct_boundary_conditions()
    alpha_phi10 = nfb.upwind_flux(alpha1, phi)
    alpha_phi10.name = "alphaPhi10"

    for a_corr in range(n_alpha_corr):
        alpha_phi_un = nfb.alpha_phase_flux(rt, alpha1, phi, c_alpha)
        alpha_phi_corr = alpha_phi_un - alpha_phi10
        # First corrector: full correction. Later correctors under-relax the
        # field against its pre-correction state (relaxOld=0.5) and add half the
        # correction flux — matching alphaEqn.H's aCorr>0 branch.
        relax_old = 0.0 if a_corr == 0 else 0.5
        nfb.mules_correct(alpha1, alpha_phi_corr, rt.dt, n_limiter, relax_old)
        alpha1.correct_boundary_conditions()
        if a_corr == 0:
            alpha_phi10 = alpha_phi10 + alpha_phi_corr
        else:
            alpha_phi10 = alpha_phi10 + 0.5 * alpha_phi_corr

    return alpha_phi10


# ---------------------------------------------------------------------------
# Operation: one MULES alpha step
# ---------------------------------------------------------------------------


@alpha_advection_model.operation(operation_number="2.0")
def alpha_advection(
    alpha1: Any,
    phi: Any,
    rho: Any,
    mu: Any,
    rhoPhi: Any,
    phase: Annotated[dict[str, float], "models"],
    surf_interp: Annotated[Any, "models"],
    alpha_settings: Annotated[dict[str, Any], "models"],
    neon_runtime: Annotated[Any, "models"],
) -> FieldUpdates:
    """One MULES-limited phase-fraction step; rebuild rho / mu / rhoPhi.

    Port of interFoam's ``alphaEqn.H`` (single corrector, no MULESCorr): rotate
    old times (so ``oldTime(rho)`` feeds the momentum ``ddt(rho,U)`` and
    ``oldTime(alpha1)`` feeds the conservative advance), build the high-order
    phase flux ``alphaPhiUn = nfb.alpha_phase_flux`` (Gauss vanLeer + ``cAlpha``
    interface compression), then let the NeoN-core FCT limiter limit the flux in
    place AND advance ``alpha1`` conservatively (no clamp). Rebuild ``rho`` /
    ``mu`` from the limited ``alpha1`` and rebuild ``rhoPhi`` from the LIMITED
    alpha flux: ``rhoPhi = alphaPhi*(rho1-rho2) + phi*rho2``. ``cAlpha`` /
    ``nLimiterIter`` come from the ``alpha.water`` fvSolution subdict.
    """
    rt = neon_runtime
    nn.rotate_old_times(rho)
    nn.rotate_old_times(alpha1)

    c_alpha = alpha_settings["c_alpha"]
    n_limiter = alpha_settings["n_limiter_iter"]

    if alpha_settings["mules_corr"]:
        # interFoam semi-implicit MULES (MULESCorr): an implicit upwind predictor
        # advances alpha1 (bounded, its upwind flux is the FCT donor base
        # alphaPhi10), then nAlphaCorr high-order correctors FCT-limit the
        # antidiffusive correction (alphaPhiUn - alphaPhi10) and accumulate it.
        alpha_phi10 = _mules_corr_solve(rt, alpha1, phi, c_alpha, n_limiter,
                                        alpha_settings["n_alpha_corr"])
    else:
        # Explicit MULES: high-order flux (vanLeer + cAlpha compression) advanced
        # once by the FCT limiter. cAlpha=0 falls back to plain vanLeer.
        alpha_phi10 = nfb.alpha_phase_flux(rt, alpha1, phi, c_alpha)
        alpha_phi10.name = "alphaPhi"
        nfb.mules_explicit_solve(
            alpha1, phi, alpha_phi10, rt.dt, 1.0, 0.0, n_limiter
        )
        alpha1.correct_boundary_conditions()
    alpha_phi = alpha_phi10

    rho1, rho2 = phase["rho1"], phase["rho2"]
    nfb.update_mixture_density(rho, alpha1, rho1, rho2)
    nfb.update_mixture_viscosity(
        mu, alpha1, rho1, rho2, phase["nu1"], phase["nu2"]
    )
    # Rebuild rhoPhi from the LIMITED alpha flux (conservation-consistent with
    # ddt(rho,U)): rhoPhi = alphaPhi*(rho1-rho2) + phi*rho2.
    rho_phi_new = alpha_phi * (rho1 - rho2) + rho2 * phi
    rhoPhi.assign(rho_phi_new)
    rho.correct_boundary_conditions()
    mu.correct_boundary_conditions()

    return FieldUpdates(
        {"alpha1": alpha1, "rho": rho, "mu": mu, "rhoPhi": rhoPhi}
    )


# ---------------------------------------------------------------------------
# Operation collection: expose alpha_advection
# ---------------------------------------------------------------------------


@alpha_advection_model.operation_collection
def collected_operations(self: Any) -> Operations:
    # The collection path bypasses the spec's default operation wrapping, so
    # wrap alpha_advection with dependency resolution here (``self`` is the
    # bound runtime).
    wrapped_alpha_advection = wrap_with_dependency_resolution(
        alpha_advection, self, alpha_advection_model._dependency_resolver
    )
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=SequentialOp(wrapped_alpha_advection),
            metadata=OperationMetadata(
                op_name="alpha_advection",
                depends_on=[],
                before=[],
                shape="box",
                color="lightgreen",
            ),
        )
    )
    return model_ops
