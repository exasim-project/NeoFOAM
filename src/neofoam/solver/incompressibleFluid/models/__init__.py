# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""incompressibleFluid solver-local model plugins."""

from .incompressibleFluidModel import incompressibleFluidModel
from .pressure_velocity import PressureVelocityAlgorithm
from .boussinesq import boussinesq
from .spalartAllmaras import spalart_allmaras

__all__ = [
    "incompressibleFluidModel",
    "PressureVelocityAlgorithm",
    "boussinesq",
    "spalart_allmaras",
]
