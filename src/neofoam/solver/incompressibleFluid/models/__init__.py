# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluid solver-local model plugins.

Exposes the plugin interface, the PressureVelocityAlgorithm dispatcher, and the
optional models (boussinesq buoyancy, adaptive Courant time-stepping).
SpalartAllmaras from the source branch is intentionally not ported in this
minimal version.
"""

from .incompressibleFluidModel import incompressibleFluidModel
from .pressure_velocity import PressureVelocityAlgorithm
from .boussinesq import boussinesq
from .adaptive_time_step import adaptiveTimeStep

__all__ = [
    "incompressibleFluidModel",
    "PressureVelocityAlgorithm",
    "boussinesq",
    "adaptiveTimeStep",
]
