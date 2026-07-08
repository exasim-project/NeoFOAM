# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleVoFNeon solver-local model plugins.

Exposes the plugin interface, the PressureVelocityAlgorithmNeoN dispatcher, the
alpha-advection core model, and the courant / maxDeltaT optional models.
Importing this package is what registers the optional models with the family
(``register_with`` side effect) — ``detect_models`` finds nothing otherwise.
"""

from .incompressibleVoFNeonModel import incompressibleVoFNeonModel
from .alpha_advection.alphaAdvectionModel import alpha_advection_model
from .pressure_velocity import PressureVelocityAlgorithmNeoN
from .courant import courant
from .max_delta_t import maxDeltaT

__all__ = [
    "incompressibleVoFNeonModel",
    "alpha_advection_model",
    "PressureVelocityAlgorithmNeoN",
    "courant",
    "maxDeltaT",
]
