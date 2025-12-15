# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
3-Stage Initialization Framework

This module implements a 3-stage initialization system for solvers and models:
- LOAD: Load configuration and data from files
- RESOLVE_DEPENDENCIES: Validate and connect models (inter-model dependencies)
- BUILD: Initialize runtime structures (fields, matrices, etc.)

Usage:
    The initialization decorators are accessed via Model and Solver:

    @dataclass
    class MyModel:
        @Model.load
        def load_data(self):
            pass

        @Model.resolve_dependencies
        def connect_dependencies(self, registry):
            pass

        @Model.build
        def initialize_fields(self, mesh):
            pass

    @dataclass
    class MySolver:
        @Solver.load
        def load_config(self):
            pass

        @Solver.resolve_dependencies
        def validate(self, registry):
            pass

        @Solver.build
        def create_context(self, mesh):
            pass
"""

from .stages import InitializationStage
from .decorators import load, resolve_dependencies, build
from .configurable import Configurable
from .config_context import ConfigContext
from .context_builder import ContextBuilder
from .initializer import SolverInitializer

__all__ = [
    "InitializationStage",
    "load",
    "resolve_dependencies",
    "build",
    "AdaptableField",
    "ModelRegistry",
    "ContextBuilder",
    "SolverInitializer",
]
