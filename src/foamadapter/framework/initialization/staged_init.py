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

from dataclasses import dataclass, field
from typing import Any, Callable

from foamadapter.framework.context import Context
from .config_context import ConfigContext
from .execution import execute_initialization
from .lazy_init import LazyInit


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

    def __init__(self, name: str, argv: list[str] | None = None):
        """
        Initialize the staged builder.

        Args:
            name: Name of the solver/model
            argv: Command-line arguments (for OpenFOAM initialization)
        """
        self.name = name
        self.argv = argv or []

        # Stage functions
        self._load_func: Callable[[], dict[str, Any]] | None = None
        self._resolve_func: Callable[[ConfigContext], None] | None = None
        self._build_func: Callable[[], list[LazyInit]] | None = None

        # Validation functions
        self._validate_load_func: Callable[[], list[ValidationError]] | None = None
        self._validate_resolve_func: (
            Callable[[ConfigContext], list[ValidationError]] | None
        ) = None

        # State storage
        self.data: Any = None  # For storing InitializationData or similar

    def load(
        self, func: Callable[[], dict[str, Any] | LoadResult]
    ) -> Callable[[], dict[str, Any] | LoadResult]:
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
        func: Callable[[list, list, ConfigContext], None]
        | Callable[[ConfigContext], None],
    ) -> Callable[[list, list, ConfigContext], None] | Callable[[ConfigContext], None]:
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
        func: Callable[[list, list], list[LazyInit]] | Callable[[], list[LazyInit]],
    ) -> Callable[[list, list], list[LazyInit]] | Callable[[], list[LazyInit]]:
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

    def validate_load(
        self, func: Callable[[], list[ValidationError]]
    ) -> Callable[[], list[ValidationError]]:
        """
        Optional decorator for LOAD stage validation.

        Usage:
            @init.validate_load
            def check_load() -> list[ValidationError]:
                errors = []
                if algo is None:
                    errors.append(ValidationError("algorithm", "Missing"))
                return errors
        """
        self._validate_load_func = func
        return func

    def validate_resolve(
        self, func: Callable[[ConfigContext], list[ValidationError]]
    ) -> Callable[[ConfigContext], list[ValidationError]]:
        """
        Optional decorator for RESOLVE stage validation.

        Usage:
            @init.validate_resolve
            def check_resolve(config: ConfigContext) -> list[ValidationError]:
                errors = []
                # Check model connections
                return errors
        """
        self._validate_resolve_func = func
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
        # === STAGE 1: LOAD ===
        if self._load_func is None:
            raise RuntimeError(f"No @{self.name}.load defined")

        load_result = self._load_func()

        # Handle both LoadResult and dict return types
        if isinstance(load_result, LoadResult):
            core_models = load_result.core_models
            optional_models = load_result.optional_models
            config_items = {}
        else:
            # Legacy dict return
            config_items = load_result
            core_models = []
            optional_models = []

        # Validate load stage
        if self._validate_load_func is not None:
            errors = self._validate_load_func()
            for e in errors:
                if e.severity == "error":
                    raise RuntimeError(f"Load error: {e.field}: {e.message}")
                else:
                    print(f"Warning: {e.field}: {e.message}")

        # === Build ConfigContext ===
        config = ConfigContext()
        for key, value in config_items.items():
            config.register(key, value)

        # === STAGE 2: RESOLVE ===
        if self._resolve_func is not None:
            # Check if resolve function expects models as parameters (free function)
            import inspect

            sig = inspect.signature(self._resolve_func)
            if len(sig.parameters) == 3:
                # Free function: resolve(core_models, optional_models, config)
                self._resolve_func(core_models, optional_models, config)
            else:
                # Legacy: resolve(config)
                self._resolve_func(config)

        # Validate resolve stage
        if self._validate_resolve_func is not None:
            warnings = self._validate_resolve_func(config)
            for w in warnings:
                print(f"Warning: {w.field}: {w.message}")

        # === STAGE 3: BUILD ===
        if self._build_func is None:
            raise RuntimeError(f"No @{self.name}.build defined")

        # Check if build function expects models as parameters (free function)
        import inspect

        sig = inspect.signature(self._build_func)
        if len(sig.parameters) == 2:
            # Free function: build(core_models, optional_models)
            lazy_inits = self._build_func(core_models, optional_models)
        else:
            # Legacy: build()
            lazy_inits = self._build_func()

        # === Execute ===
        ctx = execute_initialization(lazy_inits)

        return ctx

    def run_load(self) -> dict[str, Any] | LoadResult:
        """Execute only LOAD stage and return config items or LoadResult."""
        if self._load_func is None:
            raise RuntimeError(f"No @{self.name}.load defined")

        load_result = self._load_func()

        # Store models for run_resolve() and run_build()
        if isinstance(load_result, LoadResult):
            self._core_models = load_result.core_models
            self._optional_models = load_result.optional_models

        return load_result

    def run_resolve(self, config: ConfigContext) -> None:
        """Execute only RESOLVE stage."""
        if self._resolve_func is not None:
            # Check if resolve function expects models as parameters (free function)
            import inspect

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
        import inspect

        sig = inspect.signature(self._build_func)
        if len(sig.parameters) == 2:
            # Free function: build(core_models, optional_models)
            core_models = getattr(self, "_core_models", [])
            optional_models = getattr(self, "_optional_models", [])
            return self._build_func(core_models, optional_models)
        else:
            # Legacy: build()
            return self._build_func()
