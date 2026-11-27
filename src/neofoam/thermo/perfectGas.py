# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""perfectGas equation of state with constant Cv-or-Cp and const-or-sutherland transport.

Matches classic OpenFOAM's ``hePsiThermo``/``pureMixture``/``perfectGas``/
``hConst``-or-``eConst``/``const``-or-``sutherland`` thermoType, read from
``constant/thermophysicalProperties``.
"""

from __future__ import annotations

from typing import Any

import neon._neon as nn

from neofoam import neofoam_bindings as nfb

UNIVERSAL_GAS_CONSTANT = 8314.46261815324  # J/(kmol K)


class PerfectGasTransport:
    def __init__(self, rt: Any) -> None:
        props = nfb.read_thermophysical_properties(rt)
        thermo_type = nfb.read_thermo_type(rt)
        self.transport_type = thermo_type["transport"]
        self.energy_type = thermo_type["energy"]
        if self.energy_type not in ("sensibleInternalEnergy", "sensibleEnthalpy"):
            raise ValueError(
                f"PerfectGasTransport: unsupported energy type {self.energy_type!r} "
                "(need sensibleInternalEnergy or sensibleEnthalpy)"
            )
        # he = Cv*(T - Tref) for sensibleInternalEnergy, h = Cp*(T - Tref) for
        # sensibleEnthalpy -- OpenFOAM's eConst/hConst thermo both reduce to this exact
        # affine form for perfectGas (H(p,T) = 0, Hsref = 0 by default).
        self.is_enthalpy = self.energy_type == "sensibleEnthalpy"
        self.energy_field_name = "h" if self.is_enthalpy else "he"

        self.R = UNIVERSAL_GAS_CONSTANT / props["molWeight"]
        self.Cv = props["Cv"] if "Cv" in props else props["Cp"] - self.R
        self.Cp = self.Cv + self.R
        self.gamma = self.Cp / self.Cv
        self.energy_coeff = self.Cp if self.is_enthalpy else self.Cv

        if self.transport_type == "const":
            self.mu_value = props["mu"]
            self.Pr = props["Pr"]
            # alpha = mu/Pr = kappa/Cp: the diffusivity for *temperature* diffusion
            # (laplacian(alpha, T) represents div(kappa*grad(T))/Cp). Diffusing he
            # (internal energy) instead needs div(kappa*grad(T)) directly, i.e.
            # kappa/Cv -- matching OpenFOAM's own heThermo::alphahe() =
            # CpByCpv()*alpha_ = gamma*(kappa/Cp) = kappa/Cv exactly.
            self.alpha = self.mu_value / self.Pr
            self.alphaHe = self.gamma * self.alpha
        elif self.transport_type == "sutherland":
            self.As = props["As"]
            self.Ts = props["Ts"]
        else:
            raise ValueError(
                f"PerfectGasTransport: unsupported transport type {self.transport_type!r} "
                "(need const or sutherland)"
            )

    def psi(self, T: Any) -> Any:
        return 1.0 / (self.R * T)

    def rho(self, psi: Any, p: Any) -> Any:
        return psi * p

    def mu(self, T: Any) -> Any:
        """Dynamic viscosity: a fixed scalar for const transport, or Sutherland's
        mu(T) = As*sqrt(T)/(1 + Ts/T) for sutherland transport (both fields and plain
        scalars work as T, since this is just field/scalar arithmetic)."""
        if self.transport_type == "sutherland":
            return self.As * nn.sqrt(T) / (1.0 + self.Ts / T)
        return self.mu_value

    def kappa(self, T: Any) -> Any:
        """Thermal conductivity. Only meaningful for sutherland transport: Eucken's
        relation kappa = mu*Cv*(1.32 + 1.77*R/Cv), matching OpenFOAM's
        sutherlandTransport::kappa() exactly for this thermo type -- NOT the Pr-based
        kappa = Cp*mu/Pr that const transport uses instead (see alphaEnergy below)."""
        return self.mu(T) * self.Cv * (1.32 + 1.77 * self.R / self.Cv)

    def alphaEnergy(self, T: Any) -> Any:
        """Diffusivity for whichever energy variable this case actually transports:
        kappa/Cv for sensibleInternalEnergy (he), kappa/Cp for sensibleEnthalpy (h).
        const transport: the fixed self.alphaHe/self.alpha (T unused, kept for a
        uniform call signature with the sutherland branch). sutherland transport:
        T-dependent via kappa(T)."""
        if self.transport_type == "sutherland":
            kappa = self.kappa(T)
            return kappa / self.Cp if self.is_enthalpy else kappa / self.Cv
        return self.alpha if self.is_enthalpy else self.alphaHe
