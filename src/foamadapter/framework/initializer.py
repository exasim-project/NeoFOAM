# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar, cast

from .initialization import load, resolve_dependencies, build
from .initialization.helpers import field as field_func
from .initialization.helpers import lazy as lazy_func
from .initialization.helpers import model as model_func
from .initialization.helpers import operator as operator_func
from .initialization.initializer import SolverInitializer
from .context import Context

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)


class InitializerNamespace(Protocol):
    """Protocol defining the Initializer decorator namespace."""

    def __call__(self, cls: C) -> C:
        """Decorate a class as an Initializer."""
        ...

    # Stage decorators
    load: Callable[[F], F]
    resolve_dependencies: Callable[[F], F]
    build: Callable[[F], F]

    # Helper functions
    field: Callable[..., Any]
    operator: Callable[..., Any]
    lazy: Callable[..., Any]
    model: Callable[..., Any]


def _Initializer(cls: C) -> C:
    """Class decorator to mark a class as a 3-stage Initializer."""
    return cls


Initializer: InitializerNamespace = cast(InitializerNamespace, _Initializer)
Initializer.load = load
Initializer.resolve_dependencies = resolve_dependencies
Initializer.build = build
Initializer.field = field_func
Initializer.operator = operator_func
Initializer.lazy = lazy_func
Initializer.model = model_func


class BaseInitializer:
    """Base class for custom initializers that provides a run() method."""

    def run(self) -> Context:
        """Execute the 3-stage initialization and return Context."""
        print(f"DEBUG: Starting BaseInitializer.run for {self.__class__.__name__}")
        initializer = SolverInitializer(self)
        ctx = initializer.initialize()
        print(f"DEBUG: Initialized context: {ctx}")
        return ctx
