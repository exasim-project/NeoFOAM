# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Model factory for FastAPI-style model definitions.

Provides instance-based Model API matching Init pattern.
"""

import inspect
from typing import Any, Callable
from functools import wraps

from .types import OperationMetadata, OpType, OperationNumber
from .operations import Operation, SequentialOp
from .dependency_resolver import DependencyResolver
from .context import Context


class ModelInstance:
    """
    FastAPI-style Model instance.

    Provides decorator methods for registering model operations and configuration.
    Pattern matches Init and Solver APIs:

        model = Model("BoussinesqModel")

        @model.config
        @dataclass
        class BoussinesqConfig:
            beta: float = 3e-3
            TRef: float = 300.0

        @model.operation(operation_number="2.5")
        def solve_energy(self, T: dict, config) -> FieldUpdates:
            # config auto-injected from @model.config
            pass
    """

    def __init__(self, name: str):
        self.name = name
        self._operations: list[tuple[Any, dict[str, Any]]] = []
        self._config_class: type | None = None
        self._config_instance: Any | None = None
        self._build_func: Callable[[], list[Any]] | None = None
        self._configure_algorithm_func: Callable[[Any], None] | None = None
        self._dependency_resolver = DependencyResolver()
        self.enabled = True

        # 3-stage initialization functions
        self._load_func: Callable[[], Any] | None = None
        self._resolve_func: Callable[..., None] | None = None
        self._detect_func: Callable[[], bool] | None = None

        # State storage for load results
        self._load_result: Any = None

    def load(self, func: Callable[[], Any]) -> Callable[[], Any]:
        """
        Decorator for LOAD stage function.

        The decorated function should load configuration from files or
        return configuration data.

        Usage:
            @model.load
            def load_config() -> ModelConfig:
                return ModelConfig.load("config.yaml")
        """
        self._load_func = func
        return func

    def resolve(self, func: Callable[..., None]) -> Callable[..., None]:
        """
        Decorator for RESOLVE stage function.

        The decorated function should validate and connect model dependencies.
        It receives the ConfigContext as parameter.

        Usage:
            @model.resolve
            def resolve_deps(config: ConfigContext) -> None:
                # Validate dependencies
                pass
        """
        self._resolve_func = func
        return func

    def build(self, func: Callable[[], list[Any]]) -> Callable[[], list[Any]]:
        """
        Decorator for BUILD stage function.

        The decorated function should return list of LazyInit objects.

        Usage:
            @model.build
            def build_fields() -> list[LazyInit]:
                return [field("T", create=...)]
        """
        self._build_func = func
        return func

    def detect(self, func: Callable[[], bool]) -> Callable[[], bool]:
        """
        Decorator for detection function.

        The decorated function should return True if model should be enabled.

        Usage:
            @model.detect
            def check_if_enabled() -> bool:
                return Path("constant/transportProperties").exists()
        """
        self._detect_func = func
        return func

    def configure_algorithm_step(
        self, func: Callable[[Any], None]
    ) -> Callable[[Any], None]:
        """Decorator to register algorithm configuration step."""
        self._configure_algorithm_func = func
        return func

    def run_load(self) -> Any:
        """
        Execute LOAD stage and store result.

        Returns:
            The configuration data returned by the load function
        """
        if self._load_func is None:
            return None

        self._load_result = self._load_func()
        return self._load_result

    def run_resolve(self, config: Any) -> None:
        """
        Execute RESOLVE stage.

        Args:
            config: ConfigContext for dependency resolution
        """
        if self._resolve_func is not None:
            # Check function signature to support both patterns:
            # resolve(config) or resolve() for models with internal state
            sig = inspect.signature(self._resolve_func)
            if len(sig.parameters) > 0:
                self._resolve_func(config)
            else:
                self._resolve_func()

    def run_build(self) -> list[Any]:
        """Execute BUILD stage and return list of LazyInit objects."""
        if self._build_func is None:
            return []

        # Call build function
        return self._build_func()

    def run_detect(self) -> bool:
        """
        Execute detection function.

        Returns:
            True if model should be enabled, False otherwise
        """
        if self._detect_func is not None:
            return self._detect_func()
        # Default to enabled if no detect function
        return True

    def old_build(self) -> list[Any]:
        """Legacy build method - Execute registered build step with Depends resolution."""
        if self._build_func:
            # Resolve Annotated[Type, Depends(...)] parameters
            import inspect
            from typing import get_type_hints, get_origin, get_args, Annotated
            from foamadapter.framework.initialization.depends import Depends

            sig = inspect.signature(self._build_func)
            kwargs = {}

            try:
                hints = get_type_hints(self._build_func, include_extras=True)

                for param_name, param in sig.parameters.items():
                    hint = hints.get(param_name)
                    if hint and get_origin(hint) is Annotated:
                        args = get_args(hint)
                        for arg in args[1:]:
                            if isinstance(arg, Depends):
                                # Call the dependency function
                                kwargs[param_name] = arg()
                                break
            except Exception:
                pass

            return self._build_func(**kwargs)
        return []

    def configure_algorithm(self, algorithm: Any) -> None:
        """Execute registered algorithm configuration step."""
        if self._configure_algorithm_func:
            self._configure_algorithm_func(algorithm)

    def config(self, cls: type) -> type:
        """
        Decorator to register model configuration class.

        Args:
            cls: Configuration class (typically a dataclass)

        Returns:
            Unmodified class

        Usage:
            @model.config
            @dataclass
            class BoussinesqConfig:
                beta: float = 3e-3
                TRef: float = 300.0
        """
        self._config_class = cls
        return cls

    def get_config(self) -> Any:
        """
        Get model configuration instance (lazy initialization).

        Used as a dependency provider:
            config: Annotated[MyConfig, Depends(model.get_config)]

        Returns:
            Configuration instance
        """
        if self._config_instance is None:
            if self._config_class is None:
                raise RuntimeError(f"No config class defined for model '{self.name}'")
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
        Decorator to register a model operation.

        Args:
            operation_number: Position in execution order (e.g., "2.5", "2.7")
            depends_on: List of operation names this depends on
            before: List of operation names this should execute before
            name: Optional name override (default: function name)
            inject_config: Auto-inject config if 'config' parameter exists

        Returns:
            Decorator function

        Usage:
            @model.operation(operation_number="2.5", depends_on=["solve_momentum"])
            def solve_energy(self, T: dict, phi: dict, config) -> FieldUpdates:
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


def Model(name: str) -> ModelInstance:
    """
    Create a new FastAPI-style Model instance.

    Args:
        name: Name of the model

    Returns:
        ModelInstance that can be used to decorate operations

    Example:
        model = Model("BoussinesqModel")

        @model.config
        @dataclass
        class BoussinesqConfig:
            beta: float = 3e-3

        @model.operation(operation_number="2.5")
        def solve_energy(self, T: dict, config) -> FieldUpdates:
            # config auto-injected
            pass
    """
    return ModelInstance(name)
