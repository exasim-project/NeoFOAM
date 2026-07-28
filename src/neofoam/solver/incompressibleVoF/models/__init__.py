# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""incompressibleVoF solver-local model plugins."""

from neofoam.fv_options import fvOptions
from neofoam.mrf import mrf

from .incompressibleVoFModel import incompressibleVoFModel

# MRF and fvOptions are shared with incompressibleFluid, so their specs live in
# ``neofoam.mrf`` / ``neofoam.fv_options`` and are joined to this solver's plugin
# family here.
mrf.register_with(incompressibleVoFModel)
fvOptions.register_with(incompressibleVoFModel)

__all__ = ["incompressibleVoFModel", "fvOptions", "mrf"]
