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
    def build_lazy() -> list[InitStep]:
        # Create lazy initializers
        return [field("U", create=...), model("transport", ...)]

    # Execute the full initialization
    init.argv = argv
    ctx = init.run()
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, cast

from neofoam.framework.context import Context
from neofoam.framework.solver_factory import SolverState

# from neofoam.io.input_validation import validate_models  # IO-coupled, not needed for tests
from .config_context import ConfigContext
from .execution import execute_initialization
from .init_step import InitStep
# from neofoam.io.strategies import BaseConfig  # IO-coupled, not needed for tests


def _dispatch_by_arity(
    func: Callable[..., Any], arg_sets: dict[int, tuple[Any, ...]]
) -> Any:
    """Call *func* with the arg-set matching its parameter count.

    ``arg_sets`` maps arity → positional args.  If the function's arity
    doesn't appear in the map a ``RuntimeError`` is raised.

    Example:
        _dispatch_by_arity(
            my_build,
            {0: (), 2: (core, opt)},
        )
    """
    n = len(inspect.signature(func).parameters)
    if n not in arg_sets:
        expected = " or ".join(str(k) for k in sorted(arg_sets))
        raise RuntimeError(f"Expected {expected} parameters, got {n}")
    return func(*arg_sets[n])


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
            if hasattr(model, "configs"):
                configs.extend(model.configs)
        return configs

    def validate(self) -> list[ValidationError]:
        """Validate all models in the load result and return a list of errors."""
        # Stub implementation - actual validation requires disk IO
        # return validate_models(self.configs)
        raise NotImplementedError(
            "validate() requires IO layer - not implemented in tests"
        )


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
        def build_lazy() -> list[InitStep]:
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
        self._resolve_func: Optional[Callable[..., None]] = None
        self._build_func: Optional[Callable[..., list[InitStep]]] = None

        # State storage
        self.data: Any = None  # For storing InitializationData or similar

        # Shared state container
        self.state = SolverState()

    @property
    def core_models(self) -> list[Any]:
        """Access core_models from state."""
        return self.state.core_models

    @core_models.setter
    def core_models(self, value: list[Any]) -> None:
        """Set core_models in state."""
        self.state.core_models = value

    @property
    def optional_models(self) -> list[Any]:
        """Access optional_models from state."""
        return self.state.optional_models

    @optional_models.setter
    def optional_models(self, value: list[Any]) -> None:
        """Set optional_models in state."""
        self.state.optional_models = value

    @property
    def configs(self) -> dict[str, Any]:
        """Access configs from state."""
        return self.state.configs

    @configs.setter
    def configs(self, value: dict[str, Any]) -> None:
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
            Callable[[list[Any], list[Any], ConfigContext], None],
            Callable[[ConfigContext], None],
        ],
    ) -> Union[
        Callable[[list[Any], list[Any], ConfigContext], None],
        Callable[[ConfigContext], None],
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
            Callable[[list[Any], list[Any]], list[InitStep]],
            Callable[[], list[InitStep]],
        ],
    ) -> Union[
        Callable[[list[Any], list[Any]], list[InitStep]],
        Callable[[], list[InitStep]],
    ]:
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

        # Wire LoadResult models into ConfigContext (P-6.2)
        config = ConfigContext()
        for m in load_result.all_models:
            key = getattr(m, "name", None) or type(m).__name__.lower()
            if config.contains(key):
                raise ValueError(f"Duplicate model registration key: '{key}'")
            config.register(key, m)

        if self._resolve_func is not None:
            _dispatch_by_arity(
                self._resolve_func,
                {1: (config,), 3: (self.core_models, self.optional_models, config)},
            )

        if self._build_func is None:
            raise RuntimeError(f"No @{self.name}.build defined")

        lazy_inits = cast(
            list[InitStep],
            _dispatch_by_arity(
                self._build_func,
                {0: (), 2: (self.core_models, self.optional_models)},
            ),
        )

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
            _dispatch_by_arity(
                self._resolve_func,
                {1: (config,), 3: (self.core_models, self.optional_models, config)},
            )

    def run_build(self) -> list[InitStep]:
        """Execute only BUILD stage and return lazy initializers."""
        if self._build_func is None:
            raise RuntimeError(f"No @{self.name}.build defined")

        return cast(
            list[InitStep],
            _dispatch_by_arity(
                self._build_func,
                {0: (), 2: (self.core_models, self.optional_models)},
            ),
        )
