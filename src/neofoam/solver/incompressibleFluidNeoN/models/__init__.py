# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluidNeoN solver-local model plugins.

Exposes the plugin interface, the PressureVelocityAlgorithmNeoN dispatcher,
and the courant / maxDeltaT optional models. Importing this package is what
registers the optional models with the family (``register_with`` side effect)
— ``detect_models`` finds nothing otherwise.
"""

from neofoam.mrf import mrfNeoN

from .courant import courant
from .incompressibleFluidNeoNModel import incompressibleFluidNeoNModel
from .max_delta_t import maxDeltaT
from .pressure_velocity import PressureVelocityAlgorithmNeoN

# Registered here rather than in neofoam.mrf: that module does not import this
# family, so the spec cannot attach itself the way the models above do.
mrfNeoN.register_with(incompressibleFluidNeoNModel)

__all__ = [
    "incompressibleFluidNeoNModel",
    "PressureVelocityAlgorithmNeoN",
    "courant",
    "maxDeltaT",
    "mrfNeoN",
]
