# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The neon backend: interFoam's MULES alpha equation on NeoN fields."""

from __future__ import annotations

from typing import Any

import numpy as np
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb
from neofoam.algorithms.mules import (
    AlphaControls,
    AlphaFluxes,
    PhaseFractions,
    VolumetricFlux,
    read_alpha_controls,
    read_alpha_ddt_off_centring,
    sub_cycled_alpha_eqn,
)
from neofoam.algorithms.mules._neon_ops import NeoNMulesOps
from neofoam.io import DictFile
from neofoam.solver.neon_runtime import ensure_neon_initialized

# The schemes the NeoN kernels hard-code, keyed by their fvSchemes entry. ddt(alpha)
# is checked every solve, by read_alpha_ddt_off_centring as on the openfoam backend.
_SUPPORTED_SCHEMES = {
    ("divSchemes", "div(phi,alpha)"): "Gauss vanLeer",
    ("divSchemes", "div(phirb,alpha)"): "Gauss linear",
    ("gradSchemes", "default"): "Gauss linear",
}


class NeoNBackend:
    """alpha and the fixed flux as NeoN fields on one executor."""

    def __init__(self, runtime: pyf.Time, alpha_name: str, executor: str) -> None:
        ensure_neon_initialized(["advect"])
        self._runtime = runtime
        self._alpha_name = alpha_name
        c_alpha, n_limiter_iter = _read_neon_controls(alpha_name)
        _check_schemes(c_alpha)
        self._neon_runtime = nfb.create_adapter_run_time(runtime, executor)
        self._alpha: Any = nfb.read_scalar_volume_field(self._neon_runtime, alpha_name)
        # The kernels derive alpha2 = 1 - alpha itself, so no alpha2 field is kept.
        self._phases = PhaseFractions(self._alpha)
        self._neon_phi: Any = nfb.create_phi(self._neon_runtime, "U")
        self._ops = NeoNMulesOps(
            runtime, self._neon_runtime, self._alpha, self._neon_phi, (c_alpha, n_limiter_iter)
        )
        # advectorVoF's phi0: the start-time flux that scale_flux modulates.
        self._neon_phi0: Any = self._neon_phi * 1.0
        # createAlphaFluxes.H's persistent phase fluxes, overwritten every corrector.
        self._fluxes = AlphaFluxes(self._neon_phi * 0.0, self._neon_phi * 0.0)

    def alpha(self) -> np.ndarray:
        return np.asarray(self._alpha.internal_vector().copy_to_host()).copy()

    def cell_volumes(self) -> np.ndarray:
        return np.asarray(self._neon_runtime.mesh.V()).copy()

    def courant_numbers(self, delta_t: float) -> tuple[float, float]:
        """Flow and interface Courant numbers, computed on the executor."""
        co_num, alpha_co_num = nfb.courant_numbers(self._alpha, self._neon_phi, delta_t)
        return co_num, alpha_co_num

    def write(self) -> None:
        nfb.write_scalar_field(self._alpha, self._neon_runtime)

    def scale_flux(self, factor: float) -> None:
        """``phi = phi0*factor`` on the executor (no host copy)."""
        self._neon_phi.assign(self._neon_phi0 * factor)

    def advance(self) -> None:
        """``alphaEqnSubCycle.H`` minus rho/rhoPhi, at the runtime's current deltaT."""
        controls = AlphaControls(*read_alpha_controls(self._alpha_name))
        oc_coeff = read_alpha_ddt_off_centring(self._neon_runtime.mesh, controls.n_alpha_sub_cycles)
        flux = VolumetricFlux(self._neon_phi, self._ops.crank_nicolson_flux(oc_coeff))
        for _ in sub_cycled_alpha_eqn(self._ops, self._phases, flux, controls, self._fluxes):
            pass


def _read_neon_controls(alpha_name: str) -> tuple[float, int]:
    """``cAlpha`` (required) and ``nLimiterIter`` (MULES default 3) of the alpha solver dict."""
    solvers = pyf.dictionary.read("system/fvSolution").subDict("solvers")
    alpha_dict = solvers.subDict(alpha_name)
    return (
        float(alpha_dict.get[float]("cAlpha")),
        int(alpha_dict.getOrDefault[int]("nLimiterIter", 3)),
    )


def _selected_scheme_key(schemes: DictFile, key: tuple[str, str]) -> tuple[str, str]:
    """The entry OpenFOAM reads for ``key``: ``nHat`` overrides the default gradient."""
    if key == ("gradSchemes", "default") and schemes.found(("gradSchemes", "nHat")):
        return ("gradSchemes", "nHat")
    return key


def _check_schemes(c_alpha: float) -> None:
    """Refuse a case whose schemes differ from the ones the NeoN kernels implement."""
    schemes = DictFile("system/fvSchemes")
    for key, supported in _SUPPORTED_SCHEMES.items():
        if key == ("divSchemes", "div(phirb,alpha)") and c_alpha == 0:
            continue
        key = _selected_scheme_key(schemes, key)
        found = schemes.get[str](key) if schemes.found(key) else None
        if found != supported:
            raise NotImplementedError(
                f"neon backend: {'/'.join(key)} is {found!r}, but the NeoN MULES kernels "
                f"implement only {supported!r}; use backend='openfoam' for other schemes"
            )
