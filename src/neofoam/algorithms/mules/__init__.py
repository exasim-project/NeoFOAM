# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""interFoam's MULES alpha equation, written once over backend operations (:class:`MulesOps`)."""

from .equation import (
    AlphaControls,
    AlphaFluxes,
    MulesOps,
    PhaseFractions,
    VolumetricFlux,
    alpha_eqn,
    sub_cycled_alpha_eqn,
)
from .openfoam import (
    InterfaceProtocol,
    OpenFOAMMulesOps,
    alpha_ddt_off_centring,
    alpha_phase_flux,
    alpha_sub_cycle,
    crank_nicolson_flux,
    interface_compression_velocity,
    mules_implicit_predictor,
    read_alpha_controls,
    read_alpha_ddt_off_centring,
)

__all__ = [
    "AlphaControls",
    "AlphaFluxes",
    "InterfaceProtocol",
    "MulesOps",
    "OpenFOAMMulesOps",
    "PhaseFractions",
    "VolumetricFlux",
    "alpha_ddt_off_centring",
    "alpha_eqn",
    "alpha_phase_flux",
    "alpha_sub_cycle",
    "crank_nicolson_flux",
    "interface_compression_velocity",
    "mules_implicit_predictor",
    "read_alpha_controls",
    "read_alpha_ddt_off_centring",
    "sub_cycled_alpha_eqn",
]
