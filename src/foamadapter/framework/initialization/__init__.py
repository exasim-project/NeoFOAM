# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
3-Stage Initialization Framework

This module implements a 3-stage initialization system for solvers and models:
- LOAD: Load configuration and data from files
- RESOLVE_DEPENDENCIES: Validate and connect models (inter-model dependencies)
- BUILD: Initialize runtime structures using lazy initialization with DAG resolution

Usage:
    The initialization decorators are accessed via Model and Solver:

    @dataclass
    class MyModel:
        @Model.load
        def load_data(self):
            pass

        @Model.resolve_dependencies
        def connect_dependencies(self, config):
            pass

        @Model.build
        def initialize_fields(self, mesh) -> list:
            return [
                field("U", create=lambda: create_vector_field(mesh)),
                field("p", create=lambda: create_scalar_field(mesh)),
            ]

    @dataclass
    class MySolver:
        @Solver.load
        def load_config(self):
            pass

        @Solver.resolve_dependencies
        def validate(self, config):
            pass

        @Solver.build
        def create_runtime(self, mesh) -> list:
            return [
                lazy("runtime", create=lambda: create_runtime()),
                lazy("mesh", depends_on=["runtime"], create=lambda: mesh),
            ]
"""

from .stages import InitializationStage
from .decorators import load, resolve_dependencies, build
from .configurable import Configurable
from .config_context import ConfigContext
from .lazy_init import LazyInit
from .helpers import field, operator, lazy, model
from .initializer import SolverInitializer

__all__ = [
    "InitializationStage",
    "load",
    "resolve_dependencies",
    "build",
    "Configurable",
    "ConfigContext",
    "LazyInit",
    "field",
    "operator",
    "lazy",
    "model",
    "SolverInitializer",
]
