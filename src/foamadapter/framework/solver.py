# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar, cast, runtime_checkable

from foamadapter.framework.operations import OperationCollection

from .context import Context
from .decorator import operation as operation_func
from .initialization import load, resolve_dependencies, build
from .initialization.helpers import field as field_func
from .initialization.helpers import lazy as lazy_func
from .initialization.helpers import model as model_func
from .initialization.helpers import operator as operator_func

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)


class SolverNamespace(Protocol):
    """Protocol defining the Solver decorator namespace with all helper attributes."""

    def __call__(self, cls: C) -> C:
        """Decorate a class as a Solver."""
        ...

    # Stage decorators
    load: Callable[[F], F]
    resolve_dependencies: Callable[[F], F]
    build: Callable[[F], F]

    # Operation decorator
    operation: Callable[..., Any]

    # Helper functions
    field: Callable[..., Any]
    operator: Callable[..., Any]
    lazy: Callable[..., Any]
    model: Callable[..., Any]


def _Solver(cls: C) -> C:
    """
    A class decorator to mark a class as a Solver in the framework.
    Solvers define the main simulation loop and the basic execution of operations.
    Can be extended via Models.
    """

    # Add convenience initialize method to the class
    def initialize(self: Any) -> Context:
        """
        Convenience method to run the 3-stage initialization and return a Context.

        This method creates a SolverInitializer, runs all three stages
        (LOAD, RESOLVE_DEPENDENCIES, BUILD), and returns the resulting Context.

        Returns:
            Context: The simulation context with mesh, runtime, fields, and models

        Example:
            solver = IncompressibleFluid(argv=[...], algorithm="PIMPLE")
            ctx = solver.initialize()
            solver.main_loop(ctx)
        """
        from .initialization import SolverInitializer

        initializer = SolverInitializer(self)
        return initializer.initialize()

    # Add initialize method to the decorated class
    cls.initialize = initialize  # type: ignore[attr-defined]

    return cls


# Tell mypy that Solver is a namespace with attributes
Solver = cast(SolverNamespace, _Solver)

# Attach helper functions and decorators
Solver.operation = operation_func
Solver.load = load
Solver.resolve_dependencies = resolve_dependencies
Solver.build = build
Solver.field = field_func
Solver.operator = operator_func
Solver.lazy = lazy_func
Solver.model = model_func


@runtime_checkable
class SolverInterface(Protocol):
    def operations(self, domain_name: str | None = None) -> OperationCollection: ...

    def main_loop(self, ctx: Context) -> None: ...
