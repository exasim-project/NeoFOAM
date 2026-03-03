# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
SolverSpec — immutable solver definition registered once at module import.

Stores only callables; all mutable state lives on SolverRuntime created
per instantiate().
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from neofoam.framework.base_spec import BaseSpec
from neofoam.framework.context import Context
from neofoam.framework.operations import Operation

from .runtime import SolverRuntime


class SolverSpec(BaseSpec):
    """
    Immutable solver definition. Read-only after module import.

    Decorator methods store callables only; execution is delegated to
    SolverRuntime so that multiple runtimes can coexist independently.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._initialize_func: Optional[Callable[..., Context]] = None
        self._execution_graph_func: Optional[Callable[..., tuple[Any, Any]]] = None

    # ------------------------------------------------------------------
    # Decorator API (config/operation inherited from BaseSpec)
    # ------------------------------------------------------------------

    def initializer(self, func: Callable[..., Context]) -> Callable[..., Context]:
        """Decorator to register solver initializer with dependency injection support."""
        self._initialize_func = func
        return func

    def execution_graph_step(
        self, func: Callable[..., tuple[Any, Any]]
    ) -> Callable[..., tuple[Any, Any]]:
        """Decorator to register execution graph construction step."""
        self._execution_graph_func = func
        return func

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def instantiate(self, argv: Optional[list[Any]] = None) -> SolverRuntime:
        """
        Create a fresh SolverRuntime for one run of this solver.

        Args:
            argv: Optional command-line arguments for initialization.

        Returns:
            A new SolverRuntime bound to this spec.
        """
        return SolverRuntime(
            spec=self,
            name=self.name,
            argv=argv or [],
        )

    # ------------------------------------------------------------------
    # Execution helpers (called by SolverRuntime)
    # ------------------------------------------------------------------

    def _run_initialize(self, runtime: SolverRuntime) -> Context:
        """Execute the registered initialization step with dependency injection."""
        if self._initialize_func is None:
            raise RuntimeError(
                f"No initialize function registered for solver {self.name}"
            )

        # Resolve dependencies without Context (for standalone functions)
        kwargs = self._dependency_resolver.resolve_arguments(
            self._initialize_func, None
        )

        # Inject self=runtime if the function expects it
        sig = inspect.signature(self._initialize_func)
        if "self" in sig.parameters and "self" not in kwargs:
            kwargs["self"] = runtime

        # Set argv on StagedInit if it was injected as the first dependency
        first_param = next(iter(kwargs.values()), None) if kwargs else None
        if first_param is not None and hasattr(first_param, "argv"):
            first_param.argv = runtime.argv

        ctx = self._initialize_func(**kwargs)

        # Transfer state from injected StagedInit to runtime
        if first_param is not None:
            if hasattr(first_param, "state"):
                runtime.state = first_param.state
            else:
                # StagedInit stores core_models / optional_models directly
                if hasattr(first_param, "core_models"):
                    runtime.state.core_models = first_param.core_models
                if hasattr(first_param, "optional_models"):
                    runtime.state.optional_models = first_param.optional_models

        return ctx

    def _run_execution_graph(
        self, runtime: SolverRuntime, domain_name: Optional[str] = None
    ) -> tuple[Any, Any]:
        """Execute the registered execution graph step."""
        if self._execution_graph_func is None:
            raise RuntimeError(
                f"No execution_graph function registered for solver {self.name}"
            )

        sig = inspect.signature(self._execution_graph_func)
        kwargs: dict[str, Any] = {}

        if "self" in sig.parameters:
            kwargs["self"] = runtime
        if "domain_name" in sig.parameters:
            kwargs["domain_name"] = domain_name

        return self._execution_graph_func(**kwargs)

    # ------------------------------------------------------------------
    # Operation building (called by SolverRuntime.operations)
    # ------------------------------------------------------------------

    def _build_operations_for(self, runtime: SolverRuntime) -> list[Operation]:  # type: ignore[override]
        """Delegate to BaseSpec._build_operations_for."""
        return super()._build_operations_for(runtime)


def Solver(name: str) -> SolverSpec:
    """
    Create a new SolverSpec instance.

    Args:
        name: Name of the solver

    Returns:
        SolverSpec that can be used to decorate operations

    Example:
        solver = Solver("SimpleSolver")

        @solver.operation(operation_number="1.0")
        def solve_momentum(self, field1: float) -> FieldUpdates:
            pass
    """
    return SolverSpec(name)
