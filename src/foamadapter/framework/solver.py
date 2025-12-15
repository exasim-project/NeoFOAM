# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from typing import Protocol, runtime_checkable

from foamadapter.framework.operations import OperationCollection

from .context import Context
from .decorator import operation
from .initialization import load, resolve_dependencies, build
from .initialization.helpers import field, operator, lazy, model


def Solver(cls: type) -> type:
    """
    A class decorator to mark a class as a Solver in the framework.
    Solvers define the main simulation loop and the basic execution of operations.
    Can be extended via Models.
    """

    # Add convenience initialize method to the class
    def initialize(self) -> Context:
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
    cls.initialize = initialize

    return cls


Solver.operation = staticmethod(operation)  # type: ignore[attr-defined]
Solver.load = staticmethod(load)  # type: ignore[attr-defined]
Solver.resolve_dependencies = staticmethod(resolve_dependencies)  # type: ignore[attr-defined]
Solver.build = staticmethod(build)  # type: ignore[attr-defined]
Solver.field = staticmethod(field)  # type: ignore[attr-defined]
Solver.operator = staticmethod(operator)  # type: ignore[attr-defined]
Solver.lazy = staticmethod(lazy)  # type: ignore[attr-defined]
Solver.model = staticmethod(model)  # type: ignore[attr-defined]
Solver.build = staticmethod(build)  # type: ignore[attr-defined]


@runtime_checkable
class SolverInterface(Protocol):
    def operations(self, domain_name: str | None = None) -> OperationCollection: ...

    def main_loop(self, ctx: Context) -> None: ...
