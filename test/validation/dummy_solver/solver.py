# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Dummy solver init — demonstrates register_core_models + plugin_interface.

Core model (core_algorithm) registered explicitly.
Optional models (wall_model, scalar_transport) discovered via plugin system.
"""

from neofoam.framework.initialization import StagedInit

from .core_algorithm import core_algorithm
from .plugin import DummySolverModel

# Import optional models so they register with the plugin system
from . import wall_model as _wall_model  # noqa: F401
from . import scalar_transport as _scalar_transport  # noqa: F401

init = StagedInit("dummy_solver", plugin_interface=DummySolverModel)
init.register_core_models([core_algorithm])
