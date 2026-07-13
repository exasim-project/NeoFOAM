# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluid solver-local model plugins.

Exposes the plugin interface, the PressureVelocityAlgorithm dispatcher,
and the boussinesq, courant, and maxDeltaT optional models. SpalartAllmaras
from the source branch is intentionally not ported in this minimal version.
Telemetry is NOT a model: it is solver lifecycle, owned by the solver
entrypoint (``incompressibleFluid.maybe_configure_telemetry`` and the
``TelemetryDictConfig`` solver config).
"""

from .incompressibleFluidModel import incompressibleFluidModel
from .pressure_velocity import PressureVelocityAlgorithm
from .boussinesq import boussinesq
from .courant import courant
from .max_delta_t import maxDeltaT

__all__ = [
    "incompressibleFluidModel",
    "PressureVelocityAlgorithm",
    "boussinesq",
    "courant",
    "maxDeltaT",
]
