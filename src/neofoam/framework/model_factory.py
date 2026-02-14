# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Model factory for FastAPI-style model definitions.

Provides instance-based Model API matching Init pattern.
"""

import inspect
from typing import Any, Callable, Optional
from functools import wraps
from pydantic import BaseModel
from neofoam.io import BaseConfig
from typing import Literal
from pathlib import Path

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
        self._build_func: Optional[Callable[[], list[Any]]] = None
        self._dependency_resolver = DependencyResolver()
        self.enabled = True

        # 3-stage initialization functions
        self._load_func: Optional[Callable[[], Any]] = None
        self._resolve_func: Optional[Callable[..., None]] = None
        self._detect_func: Optional[Callable[[], bool]] = None

        # State storage for load results
        self._load_result: Any = None

        # Multi-config support
        self._config_classes: dict[str, type] = {}
        self._config_instances: dict[str, BaseConfig] = {}

        # NEW: Config auto-discovery and injection
        self._configs: dict[str, BaseConfig] = {}  # Loaded config instances
        self._discovered_configs: dict[type, str] = {}  # config_type -> param_name
        self._explicit_configs: dict[type, dict] = {}  # config_type -> {name, loader}

        # Model instance state (for operation counters, etc.)
        self._step1_count: int = 0
        self._step2_count: int = 0

    @property
    def configs(self) -> list[BaseConfig]:
        """
        Get all configs for this model as a list (backward compatibility).

        Returns:
            List of config instances
        """
        return list(self._configs.values())

    def get_configs_dict(self) -> dict[str, BaseConfig]:
        """
        Get all configs for this model as a dictionary.

        Returns:
            Dictionary mapping config names to config instances
        """
        return self._configs

    def config(self, name: Optional[str] = None) -> BaseConfig:
        """
        Get config by name, or single config if only one exists.

        Args:
            name: Config name, or None to get single config

        Returns:
            Config instance

        Raises:
            ValueError: If name is None but multiple configs exist
            KeyError: If named config not found
        """
        if name is None:
            if len(self._configs) == 0:
                raise ValueError(f"Model {self.name} has no configs loaded")
            if len(self._configs) == 1:
                return list(self._configs.values())[0]
            raise ValueError(
                f"Model {self.name} has {len(self._configs)} configs, must specify name. "
                f"Available: {list(self._configs.keys())}"
            )
        if name not in self._configs:
            raise KeyError(
                f"Config '{name}' not found in model {self.name}. "
                f"Available: {list(self._configs.keys())}"
            )
        return self._configs[name]

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def _discover_configs_from_signature(self, func: Callable) -> list[dict]:
        """
        Introspect function signature to discover BaseConfig parameters.

        Args:
            func: Operation function to introspect

        Returns:
            List of dicts with 'param_name' and 'config_type'
        """
        sig = inspect.signature(func)
        discovered = []

        for param_name, param in sig.parameters.items():
            # Skip self and ctx
            if param_name in ("self", "ctx"):
                continue

            # Check if parameter has BaseConfig type annotation
            if param.annotation != inspect.Parameter.empty:
                annotation = param.annotation

                # Handle typing constructs (e.g., Optional, Union)
                origin = getattr(annotation, "__origin__", None)
                if origin is not None:
                    continue  # Skip complex types for now

                # Check if it's a BaseConfig subclass
                try:
                    if isinstance(annotation, type) and issubclass(
                        annotation, BaseConfig
                    ):
                        discovered.append(
                            {
                                "param_name": param_name,
                                "config_type": annotation,
                            }
                        )
                except TypeError:
                    # Not a class, skip
                    continue

        return discovered

    def _create_config_injecting_wrapper(
        self,
        func: Callable,
        discovered_configs: list[dict],
    ) -> Callable:
        """
        Wrap operation function to inject configs from self._configs.

        This returns a wrapper that already has `self` and `ctx` handled,
        so dependency resolution won't try to add them again.

        Args:
            func: Original operation function
            discovered_configs: List of discovered config metadata

        Returns:
            Wrapped function that injects configs
        """
        # Check if function expects 'self' and 'ctx' parameters
        sig = inspect.signature(func)
        expects_self = "self" in sig.parameters
        expects_ctx = "ctx" in sig.parameters

        @wraps(func)
        def wrapper(ctx: Context) -> Any:
            # Build kwargs for function call
            call_kwargs = {}

            # Add ctx only if expected
            if expects_ctx:
                call_kwargs["ctx"] = ctx

            # Add self if expected
            if expects_self:
                call_kwargs["self"] = self

            # Inject discovered configs
            for cfg in discovered_configs:
                param_name = cfg["param_name"]
                config_type = cfg["config_type"]

                # Look up config in self._configs by type
                # Try exact name match first
                if param_name in self._configs:
                    call_kwargs[param_name] = self._configs[param_name]
                else:
                    # Search by type
                    found = False
                    for config_name, config_instance in self._configs.items():
                        if isinstance(config_instance, config_type):
                            call_kwargs[param_name] = config_instance
                            found = True
                            break
                    if not found:
                        raise ValueError(
                            f"Config of type {config_type.__name__} not found in model {self.name}. "
                            f"Available configs: {list(self._configs.keys())}"
                        )

            # Inject fields from ctx for non-ctx parameters
            for param_name, param in sig.parameters.items():
                # Skip already handled parameters
                if param_name in ["self", "ctx"] or param_name in call_kwargs:
                    continue

                # Check if it's a simple type (float, int, str, etc.) - likely a field
                if (
                    param.annotation in [float, int, str, bool]
                    or param.annotation == inspect.Parameter.empty
                ):
                    # Try to get from ctx.fields
                    if param_name in ctx.fields:
                        call_kwargs[param_name] = ctx.fields[param_name]

            # Call original function and handle result
            result = func(**call_kwargs)

            # If it's a FieldUpdates, update context
            from .context import FieldUpdates

            if isinstance(result, FieldUpdates):
                ctx.fields.update(result)
                return None

            return result

        # Set a flag so dependency resolution knows not to wrap this again
        wrapper._already_wrapped = True
        return wrapper

    def _generate_auto_load(self) -> Callable[[Path], tuple]:
        """
        Generate load function from discovered and explicit configs.

        Returns:
            Load function that accepts case_dir and returns tuple of config instances
        """
        import os
        from pathlib import Path

        # Combine discovered and explicit configs
        all_config_types = set(self._discovered_configs.keys()) | set(
            self._explicit_configs.keys()
        )

        def auto_load(case_dir: Path = None) -> tuple:
            # Determine case directory if not provided
            if case_dir is None:
                config_dir_env = os.environ.get("DUMMY_SOLVER_CONFIG_DIR")
                if config_dir_env:
                    case_dir = Path(config_dir_env)
                else:
                    # Default: configs subdirectory
                    # Try to find configs directory relative to current file
                    case_dir = Path.cwd() / "configs"

            loaded = {}

            for config_type in all_config_types:
                # Determine storage name
                if config_type in self._explicit_configs and self._explicit_configs[
                    config_type
                ].get("name"):
                    param_name = self._explicit_configs[config_type]["name"]
                elif config_type in self._discovered_configs:
                    param_name = self._discovered_configs[config_type]
                else:
                    # Default to class name in snake_case
                    param_name = self._to_snake_case(config_type.__name__)

                # Load config
                if config_type in self._explicit_configs and self._explicit_configs[
                    config_type
                ].get("loader"):
                    # Custom loader
                    loader = self._explicit_configs[config_type]["loader"]
                    config_instance = loader(case_dir)
                else:
                    # Default: use ConfigClass.load() from @IOStrategy
                    config_instance = config_type.load(
                        case_dir=case_dir, validate=False
                    )

                loaded[param_name] = config_instance

            # Store in model
            self._configs = loaded
            return tuple(loaded.values())

        return auto_load

    @property
    def old_configs(self) -> list[BaseConfig]:
        """Convenience property to get all config models together."""
        return list(self._config_instances.values())

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

    def run_load(self, case_dir: Path = None) -> Any:
        """
        Execute LOAD stage and store result, auto-generating if not explicitly defined.

        Args:
            case_dir: Optional path to configuration directory

        Returns:
            The configuration data returned by the load function
        """
        # NEW: If no explicit load function, generate one from discoveries
        if self._load_func is None and (
            self._discovered_configs or self._explicit_configs
        ):
            self._load_func = self._generate_auto_load()

        if self._load_func is None:
            return None

        # Check if load function accepts parameters
        import inspect

        sig = inspect.signature(self._load_func)
        if len(sig.parameters) > 0:
            self._load_result = self._load_func(case_dir)
        else:
            self._load_result = self._load_func()

        # Auto-register loaded BaseConfig instances in _config_instances (legacy support)
        # This ensures they're available via the configs property for validation
        from neofoam.io import BaseConfig

        # Handle single config or tuple/list of configs
        configs_to_register = []
        if isinstance(self._load_result, (tuple, list)):
            configs_to_register = list(self._load_result)
        elif self._load_result is not None:
            configs_to_register = [self._load_result]

        # Register each BaseConfig instance
        for idx, config in enumerate(configs_to_register):
            if isinstance(config, BaseConfig):
                # Use registered name if available, otherwise use index-based name
                config_names = list(self._config_classes.keys())
                if idx < len(config_names):
                    name = config_names[idx]
                else:
                    name = f"config_{idx}"
                self._config_instances[name] = config

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

    def with_config(
        self,
        config_class: type,
        name: Optional[str] = None,
        loader: Optional[Callable[[Any], BaseConfig]] = None,
    ) -> "ModelInstance":
        """
        Register a configuration class with the model (chainable).

        NEW: Also stores explicit config metadata for auto-load generation.
        Use this only for custom storage names or custom loading logic.

        Args:
            config_class: Pydantic BaseModel subclass for configuration
            name: Optional name for the config storage (defaults to param name from operations)
            loader: Optional custom loader function(case_dir) -> config_instance

        Returns:
            self (for method chaining)

        Usage:
            # Basic registration (for legacy compatibility)
            model.with_config(MainConfig)

            # With custom name
            model.with_config(OpConfig, name="step_config")

            # With custom loader
            model.with_config(DbConfig, loader=lambda case_dir: load_from_db())
        """
        # Store explicit config metadata for auto-discovery
        self._explicit_configs[config_class] = {
            "name": name,
            "loader": loader,
        }

        # Legacy support: Validate that it's a BaseModel
        if not (isinstance(config_class, type) and issubclass(config_class, BaseModel)):
            raise TypeError(
                f"Config class must be a Pydantic BaseModel, got {type(config_class)}"
            )

        # Legacy support: Determine config name for _config_classes
        if name is None:
            # Default to "main" if no name provided
            name = "main"

        # Check for duplicates
        if name in self._config_classes:
            # Allow re-registration if it's the same class
            if self._config_classes[name] is not config_class:
                raise ValueError(
                    f"Config with name '{name}' already exists for model '{self.name}'"
                )
        else:
            # Register the config class (legacy)
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

    def register_with(self, plugin_interface: type) -> "ModelInstance":
        """
        Register this ModelInstance with a PluginSystem interface.

        Creates a dynamic wrapper class that registers this model instance
        with the plugin system for type-safe discovery.

        Args:
            plugin_interface: The PluginSystem interface to register with
            model_type: The discriminator value (default: lowercase model name)

        Returns:
            self (for method chaining)

        Usage:
            model1 = Model("Model1").register_with(ModelInterface, model_type="model1")
        """

        # Create dynamic wrapper class
        wrapper_class = type(
            self.name,
            (BaseModel,),
            {
                "__module__": plugin_interface.__module__,
                "__annotations__": {
                    "model_type": Literal[self.name],
                },
                "model_type": self.name,
                "get_model_instance": lambda self_wrapper: self,
            },
        )

        # Register with plugin interface
        if not hasattr(plugin_interface, "register"):
            raise TypeError(
                f"Plugin interface '{plugin_interface}' does not have a 'register' method"
            )

        plugin_interface.register(wrapper_class)

        return self

    def operation(
        self,
        operation_number: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        before: Optional[list[str]] = None,
        name: Optional[str] = None,
        inject_config: bool = True,
        configs: Optional[list[str]] = None,
    ) -> Callable:
        """
        Decorator to register a model operation with auto-config discovery.

        Args:
            operation_number: Position in execution order (e.g., "2.5", "2.7")
            depends_on: List of operation names this depends on
            before: List of operation names this should execute before
            name: Optional name override (default: function name)
            inject_config: Auto-inject config if 'config' parameter exists (legacy)
            configs: List of config names to inject (legacy, deprecated)

        Returns:
            Decorator function

        Usage:
            # NEW: Auto-discovery from type annotations
            @model.operation(operation_number="2.5")
            def solve_energy(self, ctx: Context, step_config: StepConfig) -> FieldUpdates:
                # step_config auto-discovered and injected
                pass
        """

        def decorator(func: Callable) -> Callable:
            # Auto-discover configs from type annotations
            discovered = self._discover_configs_from_signature(func)

            # Store discovered configs for later load generation
            for cfg in discovered:
                if cfg["config_type"] not in self._discovered_configs:
                    self._discovered_configs[cfg["config_type"]] = cfg["param_name"]

            # NEW: Wrap function to inject configs if discovered
            if discovered:
                func_to_wrap = self._create_config_injecting_wrapper(func, discovered)
                # Config wrapper already handles everything, skip dependency resolution
                wrapped = func_to_wrap
            else:
                # Legacy: Validate that all requested configs are registered
                if configs:
                    for config_name in configs:
                        if config_name not in self._config_classes:
                            raise ValueError(
                                f"Config '{config_name}' not registered with model '{self.name}'. "
                                f"Use model.with_config() to register it first."
                            )

                # Auto-inject config if requested (legacy single-config mode)
                if inject_config and not configs:
                    func_to_wrap = self._wrap_with_config_injection(func)
                elif configs:
                    # Inject multiple configs if specified (legacy)
                    func_to_wrap = self._wrap_with_multi_config_injection(func, configs)
                else:
                    func_to_wrap = func

                # Wrap with dependency resolution so it can be called with just Context
                wrapped = self._wrap_with_dependency_resolution(func_to_wrap)

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
