# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluidNeoN solver-local model plugins.

Exposes the plugin interface, the PressureVelocityAlgorithmNeoN dispatcher,
and the courant / maxDeltaT optional models. Importing this package is what
registers the optional models with the family (``register_with`` side effect)
— ``detect_models`` finds nothing otherwise.
"""

from .courant import courant
from .incompressibleFluidNeoNModel import incompressibleFluidNeoNModel
from .max_delta_t import maxDeltaT
from .pressure_velocity import PressureVelocityAlgorithmNeoN

__all__ = [
    "incompressibleFluidNeoNModel",
    "PressureVelocityAlgorithmNeoN",
    "courant",
    "maxDeltaT",
]
