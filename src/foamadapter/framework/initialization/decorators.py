# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Stage Decorators

Decorators for marking methods as belonging to specific initialization stages.
"""

from functools import wraps
from typing import Callable

from .stages import InitializationStage


def load(func: Callable) -> Callable:
    """
    Mark a method as belonging to LOAD stage.

    This decorator is typically accessed via Model.load or Solver.load.

    Methods marked with this decorator will be called during the LOAD
    stage of initialization, where configuration and data are loaded from files.

    Example:
        @Model.load
        def load_properties(self):
            self.config = load_from_file("properties.yaml")
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._init_stage = InitializationStage.LOAD
    return wrapper


def resolve_dependencies(func: Callable) -> Callable:
    """
    Mark a method as belonging to RESOLVE_DEPENDENCIES stage.

    This decorator is typically accessed via Model.resolve_dependencies or
    Solver.resolve_dependencies.

    Methods marked with this decorator will be called during the RESOLVE_DEPENDENCIES
    stage, where models can reference each other and perform validation.
    These methods receive the ConfigContext as an argument.

    Example:
        @Model.resolve_dependencies
        def connect_transport(self, config: ConfigContext):
            self.transport = config.get("transport")
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._init_stage = InitializationStage.RESOLVE_DEPENDENCIES
    return wrapper


def build(func: Callable) -> Callable:
    """
    Mark a method as belonging to BUILD stage.

    This decorator is typically accessed via Model.build or Solver.build.

    Methods marked with this decorator will be called during the BUILD
    stage, where runtime structures like fields and matrices are initialized.
    These methods receive the mesh as an argument.

    Example:
        @Model.build
        def initialize_fields(self, mesh):
            self.velocity_field = create_field(mesh)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._init_stage = InitializationStage.BUILD
    return wrapper
