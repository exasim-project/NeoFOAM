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
    def load_config() -> dict[str, Any]:
        # Load configuration from files
        return {"algorithm": algorithm, "config": config}

    @init.resolve
    def resolve_deps(config: ConfigContext) -> None:
        # Validate and connect models
        pass

    @init.build
    def build_lazy() -> list[LazyInit]:
        # Create lazy initializers
        return [field("U", create=...), model("transport", ...)]

    # Execute the full initialization
    init.argv = argv
    ctx = init.run()
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from foamadapter.framework.context import Context
from foamadapter.framework.solver_factory import SolverState
from foamadapter.io.input_validation import validate_models
from .config_context import ConfigContext
from .execution import execute_initialization
from .lazy_init import LazyInit
from foamadapter.io.strategies import BaseConfig


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
    def configs(self) -> list[BaseConfig]:
        """Convenience property to get all config models together."""
        configs: list[BaseConfig] = []
        for model in self.all_models:
            configs.extend(model.configs)
        return configs

    def validate(self) -> list[ValidationError]:
        """Validate all models in the load result and return a list of errors."""

        return validate_models(self.configs)


class StagedInit:
    """
    3-stage initialization builder using decorators.

    Usage:
        init = StagedInit("DummySolver")

        @init.load
        def load_config() -> dict[str, Any]:
            return {"algorithm": algo, "config": cfg}

        @init.resolve
        def resolve_deps(config: ConfigContext) -> None:
            # Connect models
            pass

        @init.build
        def build_lazy() -> list[LazyInit]:
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

        # Stage functions
        self._load_func: Optional[Callable[[], LoadResult]] = None
        self._resolve_func: Optional[Callable[[ConfigContext], None]] = None
        self._build_func: Optional[Callable[[], list[LazyInit]]] = None

        # State storage
        self.data: Any = None  # For storing InitializationData or similar

        # Shared state container
        self.state = SolverState()

    @property
    def core_models(self) -> list:
        """Access core_models from state."""
        return self.state.core_models

    @core_models.setter
    def core_models(self, value: list) -> None:
        """Set core_models in state."""
        self.state.core_models = value

    @property
    def optional_models(self) -> list:
        """Access optional_models from state."""
        return self.state.optional_models

    @optional_models.setter
    def optional_models(self, value: list) -> None:
        """Set optional_models in state."""
        self.state.optional_models = value

    @property
    def configs(self) -> dict:
        """Access configs from state."""
        return self.state.configs

    @configs.setter
    def configs(self, value: dict) -> None:
        """Set configs in state."""
        self.state.configs = value

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
        self._load_func = func
        return func

    def resolve(
        self,
        func: Union[
            Callable[[list, list, ConfigContext], None], Callable[[ConfigContext], None]
        ],
    ) -> Union[
        Callable[[list, list, ConfigContext], None], Callable[[ConfigContext], None]
    ]:
        """
        Decorator for RESOLVE stage function.

        Usage (free function):
            @init.resolve
            def resolve_deps(core_models: list, optional_models: list, config: ConfigContext) -> None:
                for model in optional_models:
                    model.resolve(config)
        """
        self._resolve_func = func
        return func

    def build(
        self,
        func: Union[
            Callable[[list, list], list[LazyInit]], Callable[[], list[LazyInit]]
        ],
    ) -> Union[Callable[[list, list], list[LazyInit]], Callable[[], list[LazyInit]]]:
        """
        Decorator for BUILD stage function.

        Usage (free function):
            @init.build
            def build_lazy(core_models: list, optional_models: list) -> list[LazyInit]:
                return [
                    field("U", create=lambda ctx: ...),
                    model("algorithm", create=lambda ctx: core_models[0]),
                ]
        """
        self._build_func = func
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
        if self._load_func is None:
            raise RuntimeError(f"No @{self.name}.load defined")

        load_result = self._load_func()

        self.core_models = load_result.core_models
        self.optional_models = load_result.optional_models
        config_items = {}

        config = ConfigContext()
        for key, value in config_items.items():
            config.register(key, value)

        if self._resolve_func is not None:
            self._resolve_func(self.core_models, self.optional_models, config)

        if self._build_func is None:
            raise RuntimeError(f"No @{self.name}.build defined")

        # Check if build function expects models as parameters (free function)
        sig = inspect.signature(self._build_func)
        if len(sig.parameters) != 2:
            raise RuntimeError(
                "Build function must take exactly 2 parameters (core_models, optional_models) for free function support"
            )
            # Free function: build(core_models, optional_models)
        lazy_inits = self._build_func(self.core_models, self.optional_models)

        # === Execute ===
        ctx = execute_initialization(lazy_inits)

        if "config" in ctx.models:
            self.configs["solver"] = ctx.models["config"]
        if "solver_config" in ctx.models:
            self.configs["solver_config"] = ctx.models["solver_config"]

        return ctx

    def run_load(self) -> LoadResult:
        """Execute only LOAD stage and return config items or LoadResult."""
        if self._load_func is None:
            raise RuntimeError(f"No @{self.name}.load defined")

        load_result = self._load_func()

        self.core_models = load_result.core_models
        self.optional_models = load_result.optional_models

        return load_result

    def run_resolve(self, config: ConfigContext) -> None:
        """Execute only RESOLVE stage."""
        if self._resolve_func is not None:
            # Check if resolve function expects models as parameters (free function)
            sig = inspect.signature(self._resolve_func)
            if len(sig.parameters) == 3:
                # Free function: resolve(core_models, optional_models, config)
                core_models = getattr(self, "_core_models", [])
                optional_models = getattr(self, "_optional_models", [])
                self._resolve_func(core_models, optional_models, config)
            else:
                # Legacy: resolve(config)
                self._resolve_func(config)

    def run_build(self) -> list[LazyInit]:
        """Execute only BUILD stage and return lazy initializers."""
        if self._build_func is None:
            raise RuntimeError(f"No @{self.name}.build defined")

        # Check if build function expects models as parameters (free function)
        sig = inspect.signature(self._build_func)
        if len(sig.parameters) == 2:
            # Free function: build(core_models, optional_models)
            core_models = getattr(self, "_core_models", [])
            optional_models = getattr(self, "_optional_models", [])
            return self._build_func(core_models, optional_models)
        else:
            # Legacy: build()
            return self._build_func()
