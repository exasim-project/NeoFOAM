# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
StagedInit - 3-stage initialization with decorators.

Implements the 3-stage initialization pattern:
- LOAD: Load configuration from files
- RESOLVE: Connect models and validate dependencies
- BUILD: Create lazy initializers for runtime objects

Usage:
    init = StagedInit("DummySolver")

    @init.load
    def load_config() -> LoadResult:
        # Load configuration from files
        return LoadResult(core_models=[algorithm], optional_models=[])

    @init.resolve
    def resolve_deps(config: ConfigContext) -> None:
        # Validate and connect models
        pass

    @init.build
    def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
        # Create lazy initializers
        return [field("U", create=...), model("transport", ...)]

    # Execute the full initialization
    init.argv = argv
    ctx = init.run()
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from neofoam.framework.context import Context
from neofoam.io import BaseConfig, validate_models

# from neofoam.io.input_validation import validate_models  # IO-coupled, not needed for tests
from .config_context import ConfigContext
from .execution import execute_initialization
from .init_step import InitStep
# from neofoam.io.strategies import BaseConfig  # IO-coupled, not needed for tests


@dataclass
class ValidationError:
    """Validation error with field and message."""

    field: str
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class LoadResult:
    """
    Result from LOAD stage - contains models returned by load().

    This enables free functions by explicitly passing models between stages.
    """

    core_models: list[Any]
    optional_models: list[Any]

    @property
    def all_models(self) -> list[Any]:
        """Convenience property to get all models together."""
        return self.core_models + self.optional_models

    @property
    def configs(self) -> list[Any]:  # Changed from BaseConfig to Any to avoid import
        """Convenience property to get all config models together."""
        configs: list[Any] = []
        for model in self.all_models:
            if isinstance(model, BaseConfig):
                configs.append(model)
            if hasattr(model, "configs"):
                configs.extend(model.configs)
        return configs

    def validate(self) -> list[Any]:
        """Validate all models in the load result and return a list of errors."""
        return validate_models(self.configs)


@dataclass
class StageHooks:
    """Registered stage callbacks owned by StagedInit decorators."""

    load: Optional[Callable[[], LoadResult]] = None
    resolve: Optional[Callable[[ConfigContext], None]] = None
    build: Optional[Callable[[list[Any], list[Any]], list[InitStep]]] = None


class StagedInit:
    """
    3-stage initialization builder using decorators.

    Usage:
        init = StagedInit("DummySolver")

        @init.load
        def load_config() -> LoadResult:
            return LoadResult(core_models=[algo], optional_models=[])

        @init.resolve
        def resolve_deps(config: ConfigContext) -> None:
            # Connect models
            pass

        @init.build
        def build_lazy(core_models: list[Any], optional_models: list[Any]) -> list[InitStep]:
            return [field("U", create=...)]

        ctx = init.run()
    """

    def __init__(self, name: str, argv: Optional[list[str]] = None):
        """
        Initialize the staged builder.

        Args:
            name: Name of the solver/model
            argv: Command-line arguments (for OpenFOAM initialization)
        """
        self.name = name
        self.argv = argv or []

        # Stage callbacks + runtime model state
        self._hooks = StageHooks()
        self.core_models: list[Any] = []
        self.optional_models: list[Any] = []

        # State storage
        self.data: Any = None  # For storing InitializationData or similar

    def load(self, func: Callable[[], LoadResult]) -> Callable[[], LoadResult]:
        """
        Decorator for LOAD stage function.

        Usage:
            @init.load
            def load_config() -> LoadResult:
                # Pure function - no parameters!
                algorithm = Algorithm.from_file(...)
                return LoadResult(
                    core_models=[algorithm],
                    optional_models=DummyModel.detect_models()
                )
        """
        self._hooks.load = func
        return func

    def resolve(
        self,
        func: Callable[[ConfigContext], None],
    ) -> Callable[[ConfigContext], None]:
        """
        Decorator for RESOLVE stage function.

        Usage (free function):
            @init.resolve
            def resolve_deps(config: ConfigContext) -> None:
                ...
        """
        self._hooks.resolve = func
        return func

    def build(
        self,
        func: Callable[[list[Any], list[Any]], list[InitStep]],
    ) -> Callable[[list[Any], list[Any]], list[InitStep]]:
        """
        Decorator for BUILD stage function.

        Usage (free function):
            @init.build
            def build_lazy(core_models: list, optional_models: list) -> list[InitStep]:
                return [
                    field("U", create=lambda ctx: ...),
                    model("algorithm", create=lambda ctx: core_models[0]),
                ]
        """
        self._hooks.build = func
        return func

    def run(self) -> Context:
        """
        Execute all 3 stages and return initialized Context.

        Stages:
        1. LOAD: Load configuration from files
        2. RESOLVE: Connect models and validate dependencies
        3. BUILD: Create lazy initializers and execute

        Returns:
            Initialized Context with all fields and models
        """
        if self._hooks.load is None:
            raise RuntimeError(f"No @{self.name}.load defined")
        load_result = self._hooks.load()
        self.core_models = load_result.core_models
        self.optional_models = load_result.optional_models

        # Wire LoadResult models into ConfigContext
        config = ConfigContext()

        for loaded_model in load_result.core_models:
            key = (
                getattr(loaded_model, "name", None)
                or type(loaded_model).__name__.lower()
            )
            if config.contains(key):
                raise ValueError(f"Duplicate model registration key: '{key}'")
            config.register(key, loaded_model)

        for runtime in load_result.optional_models:
            if config.contains(runtime.name):
                raise ValueError(
                    f"Duplicate runtime registration key: '{runtime.name}'"
                )
            config.register(runtime.name, runtime)

        if self._hooks.resolve is not None:
            self._hooks.resolve(config)

        if self._hooks.build is None:
            raise RuntimeError(f"No @{self.name}.build defined")
        lazy_inits = self._hooks.build(self.core_models, self.optional_models)

        return execute_initialization(lazy_inits)

    def run_load(self) -> LoadResult:
        """Execute only LOAD stage and return config items or LoadResult."""
        if self._hooks.load is None:
            raise RuntimeError(f"No @{self.name}.load defined")
        load_result = self._hooks.load()
        self.core_models = load_result.core_models
        self.optional_models = load_result.optional_models

        return load_result

    def run_resolve(self, config: ConfigContext) -> None:
        """Execute only RESOLVE stage."""
        if self._hooks.resolve is not None:
            self._hooks.resolve(config)

    def run_build(self) -> list[InitStep]:
        """Execute only BUILD stage and return lazy initializers."""
        if self._hooks.build is None:
            raise RuntimeError(f"No @{self.name}.build defined")
        return self._hooks.build(self.core_models, self.optional_models)
