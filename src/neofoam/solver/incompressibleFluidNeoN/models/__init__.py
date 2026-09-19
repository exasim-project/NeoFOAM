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
from .mrf import mrfNeoN
from .pressure_velocity import PressureVelocityAlgorithmNeoN

# Registered here rather than at definition time in .mrf, which keeps that spec
# module independent of the family interface — as the shared MRF spec of the two
# pybFoam families is.
mrfNeoN.register_with(incompressibleFluidNeoNModel)

__all__ = [
    "incompressibleFluidNeoNModel",
    "PressureVelocityAlgorithmNeoN",
    "courant",
    "maxDeltaT",
    "mrfNeoN",
]
