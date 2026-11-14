# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MulesOps on NeoN fields, composed from the NeoN PDE solver and the NeoFOAM VoF bindings."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, Optional

import neon._neon as nn
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb

from .equation import PhaseFractions

# The MULES controls of the alpha solver dict: Ginkgo aborts on keys it does not know.
_MULES_KEYS = (
    "MULESCorr",
    "nAlphaCorr",
    "nAlphaSubCycles",
    "cAlpha",
    "nLimiterIter",
    "icAlpha",
    "scAlpha",
    "alphaApplyPrevCorr",
)


class NeoNMulesOps:
    """Every ``alphaEqn.H`` branch on one executor: MULESCorr yes/no, Euler or CrankNicolson."""

    def __init__(
        self,
        runtime: pyf.Time,
        neon_runtime: Any,
        alpha: Any,
        phi: Any,
        controls: tuple[float, int],
    ) -> None:
        self._runtime = runtime
        self._neon_runtime = neon_runtime
        self._alpha = alpha
        self._phi = phi
        # alpha1.oldTime(): NeoN's old-time slot, which imp.ddt reads and every corrector
        # advances from.
        self._alpha_old: Any = nn.old_time(alpha)
        self._alpha_old.assign(alpha)
        self._c_alpha, self._n_limiter_iter = controls
        self._delta_t = runtime.deltaTValue()
        # A new field costs a device allocation (and ~0.2 ms on host) on every pass.
        self._phic: Any = None
        self._nhatf: Any = None
        self._copies: dict[int, Any] = {}
        self._predictors: dict[int, Any] = {}
        self._upwind_fluxes: list[Any] = []
        self._phi_old: Any = None
        self._phi_cn: Any = None
        self._phi_old_time_index: Optional[int] = None

    def crank_nicolson_flux(self, oc_coeff: float) -> Any:
        """``phiCN = cnCoeff*phi + (1 - cnCoeff)*phi.oldTime()``; ``phi`` at ``ocCoeff == 0``."""
        if oc_coeff <= 0:
            return self._phi
        if self._phi_cn is None:
            self._phi_old = self._phi * 1.0
            self._phi_cn = self._phi * 1.0
            self._phi_cn.name = "phiCN"
        # GeometricField::storeOldTimes: phi.oldTime() is phi as first seen in this time step.
        time_index = self._runtime.timeIndex()
        if time_index != self._phi_old_time_index:
            self._phi_old.assign(self._phi)
            self._phi_old_time_index = time_index
        cn_coeff = 1.0 / (1.0 + oc_coeff)
        nfb.linear_combination(cn_coeff, self._phi, 1.0 - cn_coeff, self._phi_old, self._phi_cn)
        return self._phi_cn

    def compression_velocity(self, phi: Any) -> Any:
        if self._phic is None:
            self._phic = phi * 0.0
            self._nhatf = phi * 0.0
        nfb.interface_compression_velocity(phi, self._c_alpha, out=self._phic)
        return self._phic

    def phase_flux(self, phases: PhaseFractions, phi: Any, phic: Any, out: Any) -> Any:
        alpha1 = phases.alpha1
        nfb.vanleer_flux(self._neon_runtime, alpha1, phi, out=out)
        # Without compression the nHatf gradient would be computed for nothing.
        if self._c_alpha != 0.0:
            nfb.interface_nhatf(self._neon_runtime, alpha1, out=self._nhatf)
            # phir = phic*nHatf and the sum, folded into one kernel: no field per corrector.
            nfb.add_interface_compression_flux(self._neon_runtime, alpha1, phic, self._nhatf, out)
        return out

    def implicit_upwind_predictor(self, alpha1: Any, phi_cn: Any) -> Any:
        predictor = self._predictor(phi_cn)
        # The NeoN runtime's dt is the ddt's step; the sub-step's, then back.
        delta_t = self._neon_runtime.dt
        self._neon_runtime.dt = self._delta_t
        predictor.solve()
        self._neon_runtime.dt = delta_t
        # fvMatrix::solve corrects the boundary values; NeoN's PDE solve does not.
        nfb.correct_boundary_conditions_with_flux(alpha1, self._phi)
        if not self._upwind_fluxes:
            self._upwind_fluxes = [phi_cn * 0.0, phi_cn * 0.0]
        # Two buffers in turn: the loop applies the previous pass's correction, which
        # it keeps in the other buffer, after this pass's upwind flux is formed.
        self._upwind_fluxes.reverse()
        flux = self._upwind_fluxes[0]
        # Equal to fvMatrix::flux() of the upwind matrix: its upwind weights are 0 or 1.
        nfb.upwind_flux(alpha1, phi_cn, out=flux)
        return flux

    def explicit_solve(self, alpha1: Any, phi_cn: Any, alpha_phi: Any) -> None:
        nfb.mules_explicit_solve(
            alpha1,
            self._alpha_old,
            phi_cn,
            alpha_phi,
            self._delta_t,
            n_limiter_iter=self._n_limiter_iter,
        )

    def correct(self, alpha1: Any, alpha_phi: Any, alpha_phi_corr: Any) -> None:
        nfb.mules_correct(
            alpha1, alpha_phi, alpha_phi_corr, self._delta_t, n_limiter_iter=self._n_limiter_iter
        )
        nfb.correct_boundary_conditions_with_flux(alpha1, self._phi)

    def copy(self, field: Any) -> Any:
        # One buffer per source field: the loop copies only its persistent fields.
        buffer = self._copies.get(id(field))
        if buffer is None:
            buffer = self._copies[id(field)] = field * 1.0
            return buffer
        buffer.assign(field)
        return buffer

    def linear_combination(self, a: float, x: Any, b: float, y: Any, out: Any) -> None:
        nfb.linear_combination(a, x, b, y, out)

    def correct_interface(self) -> None:
        # nHatf is computed from the current alpha on every call: no cached state.
        return None

    def info(self, message: str) -> None:
        pyf.Info(message)

    def sub_steps(self, n_sub_cycles: int) -> Iterator[float]:
        self._delta_t = self._runtime.deltaTValue() / n_sub_cycles
        for _ in range(n_sub_cycles):
            self._alpha_old.assign(self._alpha)
            yield 1.0 / n_sub_cycles

    def _predictor(self, phi_cn: Any) -> Any:
        """``fvm::ddt(alpha1) + fvm::div(phi_cn, alpha1)``, Euler and upwind, once per flux."""
        predictor = self._predictors.get(id(phi_cn))
        if predictor is None:
            # Workaround: the bindings share one out-of-line host loop of NeoN::fill
            # whose nvcc lambda wrapper only create_uniform_volume_field's fill sets
            # up; until that has run once, the linear-system reset inside solve()
            # calls through a null pointer (SIGSEGV). The solvers happen to create
            # nu that way first.
            nfb.create_uniform_volume_field(self._neon_runtime, "fillWarmUp", 0.0)
            _register_predictor_settings(self._neon_runtime, self._alpha.name, phi_cn.name)
            predictor = nfb.PDESolverScalar(
                nn.imp.ddt(self._alpha) + nn.imp.div(phi_cn, self._alpha),
                self._alpha,
                self._neon_runtime,
            )
            self._predictors[id(phi_cn)] = predictor
        return predictor


def _register_predictor_settings(neon_runtime: Any, alpha_name: str, flux_name: str) -> None:
    """The predictor's schemes, which alphaEqn.H hard-codes, and its Ginkgo solver dict."""
    schemes = neon_runtime.fv_schemes_dict
    # NeoN's name for Euler; ddt(alpha) itself may be CrankNicolson, which phiCN carries.
    schemes.subDict("ddtSchemes").insert_string(f"ddt({alpha_name})", "BDF1")
    schemes.subDict("divSchemes").insert_token_list(
        f"div({flux_name},{alpha_name})", nn.TokenList(["Gauss", "upwind"])
    )
    solvers = neon_runtime.fv_solution_dict.subDict("solvers")
    # The case's entry is usually a regex key ("alpha.water.*"); the exact key wins
    # NeoN's lookup. Mapping an already mapped entry returns it unchanged.
    settings = nfb.map_fv_solution(solvers.subDict(_solver_key(solvers, alpha_name)))
    for key in _MULES_KEYS:
        if settings.contains(key):
            settings.remove(key)
    solvers.insert_dict(alpha_name, settings)


def _solver_key(solvers: Any, field_name: str) -> str:
    """The ``solvers`` key selecting ``field_name``: itself, or a quoted regex OpenFOAM matches."""
    if solvers.contains(field_name):
        return field_name
    for key in solvers.keys():
        if len(key) > 1 and key[0] == key[-1] == '"' and re.fullmatch(key[1:-1], field_name):
            return str(key)
    raise KeyError(f"system/fvSolution/solvers has no entry for {field_name!r}")
