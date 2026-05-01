# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Dummy solver — demonstrates config discovery via register_core_models + plugin system.

Structure:
- plugin.py              — DummySolverModel plugin interface
- solver.py              — StagedInit with register_core_models + plugin_interface
- configs.py             — Solver-core configs (TimeConfig, FluidPropertiesConfig)
- core_algorithm.py      — Core model: all 6 scheme sections (registered explicitly)
- wall_model.py          — Optional: non-standard section wallDist (via plugin)
- scalar_transport.py    — Optional: new transported scalar field T (via plugin)
"""

from .configs import FluidPropertiesConfig, TimeConfig
from .core_algorithm import CoreAlgorithmConfig, core_algorithm, momentum, pressure_correction
from .wall_model import WallModelConfig, wall_model, wall_solve
from .scalar_transport import ScalarTransportConfig, scalar_transport, energy_equation

__all__ = [
    "TimeConfig",
    "FluidPropertiesConfig",
    "CoreAlgorithmConfig",
    "core_algorithm",
    "momentum",
    "pressure_correction",
    "WallModelConfig",
    "wall_model",
    "wall_solve",
    "ScalarTransportConfig",
    "scalar_transport",
    "energy_equation",
]
