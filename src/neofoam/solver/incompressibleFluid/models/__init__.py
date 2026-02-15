# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""incompressibleFluid solver-local model plugins."""

from .incompressibleFluidModel import incompressibleFluidModel
from .pressure_velocity import pressure_velocity
from .boussinesq import boussinesq

__all__ = [
    "incompressibleFluidModel",
    "pressure_velocity",
    "boussinesq",
]
