# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Solver factory for FastAPI-style solver definitions.

Provides instance-based Solver API matching Init pattern.
"""

import inspect
from typing import Any, Callable
from functools import wraps

from .types import OperationMetadata, OpType, OperationNumber
from .operations import Operation, SequentialOp
from .dependency_resolver import DependencyResolver
from .context import Context


class SolverInstance:
    """
    FastAPI-style Solver instance.

    Provides decorator methods for registering solver operations and configuration.
    Pattern matches Init API:

        solver = Solver("MySolver")

        @solver.config
        @dataclass
        class MySolverConfig:
            tolerance: float = 1e-6

        @solver.operation(operation_number="1.0")
        def solve_momentum(self, U: dict, p: dict, config):
            # config auto-injected from @solver.config
            pass
    """

    def __init__(self, name: str):
        self.name = name
        self._operations: list[tuple[Any, dict[str, Any]]] = []
        self._config_class: type | None = None
        self._config_instance: Any | None = None
        self._initialize_func: Callable[[], Context] | None = None
        self._execution_graph_func: Callable[[str | None], tuple[Any, Any]] | None = (
            None
        )
        self._dependency_resolver = DependencyResolver()

    def initialize_step(self, func: Callable[[], Context]) -> Callable[[], Context]:
        """Decorator to register solver initialization step."""
        self._initialize_func = func
        return func

    def execution_graph_step(
        self, func: Callable[[str | None], tuple[Any, Any]]
    ) -> Callable[[str | None], tuple[Any, Any]]:
        """Decorator to register execution graph construction step."""
        self._execution_graph_func = func
        return func

    def initialize(self) -> Context:
        """Execute registered initialization step."""
        if self._initialize_func:
            return self._initialize_func()
        raise RuntimeError(f"No initialize function registered for solver {self.name}")

    def execution_graph(self, domain_name: str | None = None) -> tuple[Any, Any]:
        """Execute registered execution graph step."""
        if self._execution_graph_func:
            return self._execution_graph_func(domain_name)
        raise RuntimeError(
            f"No execution_graph function registered for solver {self.name}"
        )

    def config(self, cls: type) -> type:
        """
        Decorator to register solver configuration class.

        Args:
            cls: Configuration class (typically a dataclass)

        Returns:
            Unmodified class

        Usage:
            @solver.config
            @dataclass
            class MySolverConfig:
                tolerance: float = 1e-6
                max_iterations: int = 1000
        """
        self._config_class = cls
        return cls

    def get_config(self) -> Any:
        """
        Get solver configuration instance (lazy initialization).

        Used as a dependency provider:
            config: Annotated[MyConfig, Depends(solver.get_config)]

        Returns:
            Configuration instance
        """
        if self._config_instance is None:
            if self._config_class is None:
                raise RuntimeError(f"No config class defined for solver '{self.name}'")
            self._config_instance = self._config_class()
        return self._config_instance

    def operation(
        self,
        operation_number: str | None = None,
        depends_on: list[str] | None = None,
        before: list[str] | None = None,
        name: str | None = None,
        inject_config: bool = True,
    ) -> Callable:
        """
        Decorator to register a solver operation.

        Args:
            operation_number: Position in execution order (e.g., "1.0", "2.3")
            depends_on: List of operation names this depends on
            before: List of operation names this should execute before
            name: Optional name override (default: function name)
            inject_config: Auto-inject config if 'config' parameter exists

        Returns:
            Decorator function

        Usage:
            @solver.operation(operation_number="1.0")
            def solve_momentum(self, U: dict, p: dict, config) -> FieldUpdates:
                # config auto-injected if inject_config=True
                pass
        """

        def decorator(func: Callable) -> Callable:
            # Auto-inject config if requested
            if inject_config:
                func = self._wrap_with_config_injection(func)

            # Wrap with dependency resolution so it can be called with just Context
            wrapped = self._wrap_with_dependency_resolution(func)

            # Store operation registration
            self._operations.append(
                (
                    wrapped,
                    {
                        "operation_number": operation_number,
                        "depends_on": depends_on,
                        "before": before,
                        "name": name or func.__name__,
                    },
                )
            )

            # Add to instance for easy access
            setattr(self, name or func.__name__, wrapped)

            # Attach metadata to function for compatibility with existing framework
            wrapped._metadata = OperationMetadata(
                op_name=name or func.__name__,
                op_type=OpType.OPERATION,
                operation_number=OperationNumber(operation_number)
                if operation_number
                else None,
                depends_on=depends_on,
            )

            return wrapped

        return decorator

    def _wrap_with_config_injection(self, func: Callable) -> Callable:
        """
        Wrap function to auto-inject config parameter if it exists in signature.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function that auto-injects config
        """
        sig = inspect.signature(func)

        # Check if 'config' parameter exists
        if "config" not in sig.parameters:
            return func

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Inject config if not already provided
            if "config" not in kwargs:
                kwargs["config"] = self.get_config()
            return func(*args, **kwargs)

        # Preserve original signature for dependency resolution
        wrapper.__signature__ = sig
        return wrapper

    @property
    def operations(self) -> list[Operation]:
        """
        Get list of Operation objects from decorated methods.

        Returns:
            List of Operation objects ready for DAG resolution
        """
        ops = []
        for func, metadata in self._operations:
            # Create SequentialOp
            seq_op = SequentialOp(func)

            # Create Operation with metadata
            op = Operation(
                func=seq_op,
                operation_name=metadata["name"],
                operation_number=OperationNumber(metadata["operation_number"])
                if metadata["operation_number"]
                else None,
                depends_on=metadata["depends_on"],
                before=metadata["before"],
            )
            ops.append(op)

        return ops

    def _wrap_with_dependency_resolution(
        self, func: Callable
    ) -> Callable[[Context], Any]:
        """
        Wrap function to resolve dependencies from Context.

        Args:
            func: Function to wrap

        Returns:
            Function that takes Context and resolves all dependencies
        """

        @wraps(func)
        def wrapper(ctx: Context) -> Any:
            # Resolve all arguments from context
            kwargs = self._dependency_resolver.resolve_arguments(func, ctx)

            # Check if first parameter is 'self'
            sig = inspect.signature(func)
            if "self" in sig.parameters and "self" not in kwargs:
                kwargs["self"] = self

            # Call function with resolved arguments
            result = func(**kwargs)

            # If it's a FieldUpdates, update context
            from .context import FieldUpdates

            if isinstance(result, FieldUpdates):
                ctx.fields.update(result)
                return None

            return result

        return wrapper


def Solver(name: str) -> SolverInstance:
    """
    Create a new FastAPI-style Solver instance.

    Args:
        name: Name of the solver

    Returns:
        SolverInstance that can be used to decorate operations

    Example:
        solver = Solver("SimpleSolver")

        @solver.config
        @dataclass
        class SimpleSolverConfig:
            tolerance: float = 1e-6

        @solver.operation(operation_number="1.0")
        def solve_momentum(self, U: dict, p: dict, config) -> FieldUpdates:
            # config auto-injected
            pass
    """
    return SolverInstance(name)
