# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""incompressibleVoF solver-local model plugins."""

from neofoam.fv_options import fvOptions

from .incompressibleVoFModel import incompressibleVoFModel
from .mrf import mrf

# MRF and fvOptions are shared with incompressibleFluid, so their specs cannot
# register themselves at definition time.
mrf.register_with(incompressibleVoFModel)
fvOptions.register_with(incompressibleVoFModel)

__all__ = ["incompressibleVoFModel", "fvOptions", "mrf"]
