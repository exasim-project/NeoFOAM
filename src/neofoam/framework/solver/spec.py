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

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import (
    DependencyResolver,
    wrap_with_dependency_resolution,
)
from neofoam.framework.operations import Operation, OperationCollection, SequentialOp
from neofoam.framework.types import OperationMetadata, OperationNumber

from .runtime import SolverRuntime


class SolverSpec:
    """
    Immutable solver definition. Read-only after module import.

    Decorator methods store callables only; execution is delegated to
    SolverRuntime so that multiple runtimes can coexist independently.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._operations: list[tuple[Any, dict[str, Any]]] = []
        self._config_class: Optional[type] = None
        self._initialize_func: Optional[Callable[..., Context]] = None
        self._execution_graph_func: Optional[Callable[..., tuple[Any, Any]]] = None
        self._dependency_resolver = DependencyResolver()

    # ------------------------------------------------------------------
    # Decorator API (store only, no side-effects)
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

    def config(self, cls: type) -> type:
        """
        Decorator to register solver configuration class.

        Usage:
            @solver.config
            @dataclass
            class MySolverConfig:
                tolerance: float = 1e-6
        """
        self._config_class = cls
        return cls

    def operation(
        self,
        operation_number: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        before: Optional[list[str]] = None,
        name: Optional[str] = None,
    ) -> Callable[..., Any]:
        """
        Decorator to register a solver operation.

        Stores the raw function — wrapping is deferred to _build_operations_for.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._operations.append(
                (
                    func,  # store original; wrapping happens in _build_operations_for
                    {
                        "operation_number": operation_number,
                        "depends_on": depends_on,
                        "before": before,
                        "name": name or func.__name__,
                    },
                )
            )
            return func

        return decorator

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

    def _build_operations_for(self, runtime: SolverRuntime) -> OperationCollection:
        """
        Build OperationCollection with *runtime* as the ``self`` binding.

        Returns a fresh collection per call so multiple runtimes never
        share wrapper state.
        """
        from neofoam.framework.config_injection import (
            _discover_configs_from_signature,
            _create_runtime_config_wrapper,
        )

        ops = OperationCollection()
        for func, metadata in self._operations:
            discovered = _discover_configs_from_signature(func)
            if discovered:
                wrapped = _create_runtime_config_wrapper(func, discovered, runtime)
            else:
                wrapped = wrap_with_dependency_resolution(
                    func, runtime, self._dependency_resolver
                )

            op = Operation(
                func=SequentialOp(wrapped),
                metadata=OperationMetadata(
                    op_name=metadata["name"],
                    operation_number=(
                        OperationNumber(metadata["operation_number"])
                        if metadata["operation_number"]
                        else None
                    ),
                    depends_on=metadata["depends_on"] or [],
                    before=metadata["before"] or [],
                ),
            )
            ops.add(op)

        return ops


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
