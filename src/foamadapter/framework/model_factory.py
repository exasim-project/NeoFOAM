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

        model = Model("BoussinesqModel").with_config(BoussinesqConfig)

        @model.operation(operation_number="2.5", configs=["main"])
        def solve_energy(self, T: dict, main: BoussinesqConfig) -> FieldUpdates:
            # main config auto-injected
            pass
    """

    def __init__(self, name: str):
        self.name = name
        self._operations: list[tuple[Any, dict[str, Any]]] = []
        self._build_func: Callable[[], list[Any]] | None = None
        self._dependency_resolver = DependencyResolver()
        self.enabled = True

        # 3-stage initialization functions
        self._load_func: Callable[[], Any] | None = None
        self._resolve_func: Callable[..., None] | None = None
        self._detect_func: Callable[[], bool] | None = None

        # State storage for load results
        self._load_result: Any = None

        # Multi-config support
        self._config_classes: dict[str, type] = {}
        self._config_instances: dict[str, Any] = {}

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

    def with_config(self, config_class: type, name: str | None = None) -> "ModelInstance":
        """
        Register a configuration class with the model (chainable).

        Args:
            config_class: Pydantic BaseModel subclass for configuration
            name: Optional name for the config (default: infer from class name or use "main")

        Returns:
            self (for method chaining)

        Usage:
            model.with_config(MainConfig).with_config(OpConfig, name="op_config")
        """
        from pydantic import BaseModel

        # Validate that it's a BaseModel
        if not (isinstance(config_class, type) and issubclass(config_class, BaseModel)):
            raise TypeError(
                f"Config class must be a Pydantic BaseModel, got {type(config_class)}"
            )

        # Determine config name
        if name is None:
            # Default to "main" if no name provided
            name = "main"

        # Check for duplicates
        if name in self._config_classes:
            raise ValueError(
                f"Config with name '{name}' already exists for model '{self.name}'"
            )

        # Register the config class
        self._config_classes[name] = config_class

        return self

    def get_config(self, name: str = "main", **kwargs: Any) -> Any:
        """
        Get or create a config instance by name.

        Args:
            name: Name of the config to retrieve (default: "main")
            **kwargs: Constructor arguments for first-time instantiation

        Returns:
            Config instance

        Usage:
            config = model.get_config("main", prop1=5.0, prop2=50.0)
            # Later calls can omit kwargs to retrieve cached instance
            config = model.get_config("main")
        """
        # Check if config class exists
        if name not in self._config_classes:
            raise KeyError(
                f"No config registered with name '{name}' for model '{self.name}'"
            )

        # Return cached instance if exists and no new kwargs provided
        if name in self._config_instances and not kwargs:
            return self._config_instances[name]

        # Create new instance
        config_class = self._config_classes[name]
        instance = config_class(**kwargs)
        self._config_instances[name] = instance

        return instance

    def operation(
        self,
        operation_number: str | None = None,
        depends_on: list[str] | None = None,
        before: list[str] | None = None,
        name: str | None = None,
        inject_config: bool = True,
        configs: list[str] | None = None,
    ) -> Callable:
        """
        Decorator to register a model operation.

        Args:
            operation_number: Position in execution order (e.g., "2.5", "2.7")
            depends_on: List of operation names this depends on
            before: List of operation names this should execute before
            name: Optional name override (default: function name)
            inject_config: Auto-inject config if 'config' parameter exists
            configs: List of config names to inject into operation parameters

        Returns:
            Decorator function

        Usage:
            @model.operation(operation_number="2.5", depends_on=["solve_momentum"])
            def solve_energy(self, T: dict, phi: dict, config) -> FieldUpdates:
                # config auto-injected if inject_config=True
                pass

            @model.operation(operation_number="2.7", configs=["main", "op_config"])
            def solve_with_configs(self, main_config, op_config) -> FieldUpdates:
                # Both configs injected
                pass
        """

        def decorator(func: Callable) -> Callable:
            # Validate that all requested configs are registered
            if configs:
                for config_name in configs:
                    if config_name not in self._config_classes:
                        raise ValueError(
                            f"Config '{config_name}' not registered with model '{self.name}'. "
                            f"Use model.with_config() to register it first."
                        )

            # Auto-inject config if requested (legacy single-config mode)
            if inject_config and not configs:
                func = self._wrap_with_config_injection(func)

            # Inject multiple configs if specified
            if configs:
                func = self._wrap_with_multi_config_injection(func, configs)

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
                if "main" in self._config_instances:
                    kwargs["config"] = self.get_config("main")
            return func(*args, **kwargs)

        # Preserve original signature for dependency resolution
        wrapper.__signature__ = sig
        return wrapper

    def _wrap_with_multi_config_injection(
        self, func: Callable, config_names: list[str]
    ) -> Callable:
        """
        Wrap function to auto-inject multiple named configs.

        Configs are injected based on parameter names in the function signature.
        The parameter name should match the config name or be a type-annotated parameter.

        Args:
            func: Function to wrap
            config_names: List of config names to inject

        Returns:
            Wrapped function that auto-injects configs
        """
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Inject configs by matching parameter names
            for config_name in config_names:
                # Try to find matching parameter
                param_name = None

                # Direct match: parameter name == config name
                if config_name in sig.parameters:
                    param_name = config_name
                else:
                    # Try to match by config suffix (e.g., "main_config" -> "main")
                    for param in sig.parameters:
                        if param.endswith("_config") or param.endswith("Config"):
                            # Extract base name
                            base = param.replace("_config", "").replace("Config", "")
                            if base == config_name or config_name.endswith(base):
                                param_name = param
                                break

                if param_name and param_name not in kwargs:
                    kwargs[param_name] = self.get_config(config_name)

            return func(*args, **kwargs)

        # Preserve original signature for dependency resolution
        wrapper.__signature__ = sig
        return wrapper

    def validate_after_load(self) -> list["ValidationError"]:
        """
        Validate all instantiated configs after LOAD stage.

        Returns:
            List of ValidationError objects (empty if validation passes)
        """
        from pydantic import ValidationError as PydanticValidationError
        from foamadapter.framework.initialization.staged_init import ValidationError

        errors: list[ValidationError] = []

        # Validate all instantiated configs
        for config_name, config_instance in self._config_instances.items():
            try:
                # Re-validate the instance (Pydantic caches validation)
                config_class = self._config_classes[config_name]
                config_class.model_validate(config_instance.model_dump())
            except PydanticValidationError as e:
                # Convert Pydantic errors to ValidationError objects
                for err in e.errors():
                    field = ".".join(str(loc) for loc in err.get("loc", []))
                    errors.append(
                        ValidationError(
                            field=f"{self.name}.{config_name}.{field}",
                            message=err.get("msg", "Validation error"),
                            severity="error",
                        )
                    )

        return errors

    def validate_after_resolve(self) -> list["ValidationError"]:
        """
        Validate all configs after RESOLVE stage.

        Currently delegates to validate_after_load(). Can be extended for
        cross-model validation logic.

        Returns:
            List of ValidationError objects (empty if validation passes)
        """
        return self.validate_after_load()

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
        model = Model("BoussinesqModel").with_config(BoussinesqConfig)

        @model.operation(operation_number="2.5", configs=["main"])
        def solve_energy(self, T: dict, main: BoussinesqConfig) -> FieldUpdates:
            # config auto-injected
            pass
    """
    return ModelInstance(name)
