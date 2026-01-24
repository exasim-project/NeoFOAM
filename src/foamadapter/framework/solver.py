# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar, cast, runtime_checkable

from foamadapter.framework.operations import OperationCollection, StepBuilder

from .context import Context
from .decorator import operation as operation_func

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)


class SolverNamespace(Protocol):
    """Protocol defining the Solver decorator namespace."""

    def __call__(self, cls: C) -> C:
        """Decorate a class as a Solver."""
        ...

    # Operation decorator
    operation: Callable[..., Any]


def _Solver(cls: C) -> C:
    """
    A class decorator to mark a class as a Solver in the framework.
    Solvers define the main simulation loop and the basic execution of operations.
    Can be extended via Models.
    """
    # @Solver decorator no longer provides a default initialize() method.
    # Solvers must implement their own explicit initialization.
    # See IncompressibleFluid for an example of the hybrid explicit approach.
    return cls


# Tell mypy that Solver is a namespace with attributes
Solver = cast(SolverNamespace, _Solver)

# Attach operation decorator
Solver.operation = operation_func


@runtime_checkable
class SolverInterface(Protocol):
    def execution_graph(
        self, domain_name: str | None = None
    ) -> tuple[StepBuilder, OperationCollection]: ...

    def operations(self, domain_name: str | None = None) -> OperationCollection: ...

    def main_loop(self, ctx: Context) -> None: ...
