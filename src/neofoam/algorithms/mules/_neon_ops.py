# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MulesOps on NeoN fields for the explicit MULES path, composed from the NeoFOAM bindings."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pybFoam as pyf

from neofoam import neofoam_bindings as nfb

from .equation import PhaseFractions


class NeoNMulesOps:
    """``MULESCorr no`` and Euler on one executor; the MULESCorr-only operations raise."""

    def __init__(
        self,
        runtime: pyf.Time,
        neon_runtime: Any,
        alpha: Any,
        c_alpha: float,
        n_limiter_iter: int,
    ) -> None:
        self._runtime = runtime
        self._neon_runtime = neon_runtime
        self._alpha = alpha
        # Every corrector advances from the start of the (sub-)step (alpha1.oldTime()).
        self._alpha_old: Any = alpha * 1.0
        self._c_alpha = c_alpha
        self._n_limiter_iter = n_limiter_iter
        self._delta_t = runtime.deltaTValue()
        # A new field costs a device allocation (and ~0.2 ms on host) on every pass.
        self._phic: Any = None
        self._nhatf: Any = None

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
        raise _mules_corr_not_wired()

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
        raise _mules_corr_not_wired()

    def copy(self, field: Any) -> Any:
        raise _mules_corr_not_wired()

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


def _mules_corr_not_wired() -> NotImplementedError:
    """The error for the MULESCorr-only operations."""
    return NotImplementedError(
        "neon backend: MULESCorr yes needs the implicit upwind predictor "
        "(fvm::ddt + fvm::div upwind), which is not wired on NeoN yet; "
        "use MULESCorr no or backend='openfoam'"
    )
