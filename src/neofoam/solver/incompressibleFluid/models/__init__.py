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

from neofoam.fv_options import fvOptions
from neofoam.mrf import mrf

from .boussinesq import boussinesq
from .courant import courant
from .incompressibleFluidModel import incompressibleFluidModel
from .max_delta_t import maxDeltaT
from .pressure_velocity import PressureVelocityAlgorithm

# MRF and fvOptions are shared with incompressibleVoF, so their specs live in
# ``neofoam.mrf`` / ``neofoam.fv_options`` and are joined to this solver's plugin
# family here (the models above register themselves at definition time, which a
# shared spec cannot do).
mrf.register_with(incompressibleFluidModel)
fvOptions.register_with(incompressibleFluidModel)

__all__ = [
    "incompressibleFluidModel",
    "PressureVelocityAlgorithm",
    "boussinesq",
    "courant",
    "fvOptions",
    "maxDeltaT",
    "mrf",
]
