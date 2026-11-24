# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleFluidBlockAMR solver-local model plugins.

Exposes the plugin interface and the ProjectionAlgorithm dispatcher. Spec 01
ships no optional models; importing this package registers the (currently empty)
family so ``detect_models`` and ``configurations()`` resolve cleanly.
"""

from .incompressibleFluidBlockAMRModel import incompressibleFluidBlockAMRModel
from .projection import ProjectionAlgorithm

__all__ = [
    "incompressibleFluidBlockAMRModel",
    "ProjectionAlgorithm",
]
