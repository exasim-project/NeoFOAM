# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""perfectGas equation of state with constant Cv-or-Cp and constant transport.

Matches classic OpenFOAM's ``hePsiThermo``/``pureMixture``/``perfectGas``/
``hConst``-or-``eConst``/``const`` thermoType, read from
``constant/thermophysicalProperties``.
"""

from __future__ import annotations

from typing import Any

from neofoam import neofoam_bindings as nfb

UNIVERSAL_GAS_CONSTANT = 8314.46261815324  # J/(kmol K)


class PerfectGasConstTransport:
    def __init__(self, rt: Any) -> None:
        props = nfb.read_thermophysical_properties(rt)
        self.R = UNIVERSAL_GAS_CONSTANT / props["molWeight"]
        self.Cv = props["Cv"] if "Cv" in props else props["Cp"] - self.R
        self.Cp = self.Cv + self.R
        self.gamma = self.Cp / self.Cv
        self.mu = props["mu"]
        self.Pr = props["Pr"]
        # alpha = mu/Pr = kappa/Cp: the diffusivity for *temperature* diffusion
        # (laplacian(alpha, T) represents div(kappa*grad(T))/Cp). Diffusing he
        # (internal energy) instead needs div(kappa*grad(T)) directly, i.e.
        # kappa/Cv -- matching OpenFOAM's own heThermo::alphahe() =
        # CpByCpv()*alpha_ = gamma*(kappa/Cp) = kappa/Cv exactly.
        self.alpha = self.mu / self.Pr
        self.alphaHe = self.gamma * self.alpha

    def psi(self, T: Any) -> Any:
        return 1.0 / (self.R * T)

    def rho(self, psi: Any, p: Any) -> Any:
        return psi * p
