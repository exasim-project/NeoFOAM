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
from neofoam.foam.verification import (
    collect_requirements_from_models,
    verify_fvschemes,
    verify_fvsolution,
    VerificationError,
)

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
    fv_schemes_config: Any = None
    fv_solution_config: Any = None

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

    def __init__(
        self,
        name: str,
        argv: Optional[list[str]] = None,
        plugin_interface: Optional[type] = None,
    ):
        """
        Initialize the staged builder.

        Args:
            name: Name of the solver/model
            argv: Command-line arguments (for OpenFOAM initialization)
            plugin_interface: PluginSystem interface for optional model discovery
        """
        self.name = name
        self.argv = argv or []

        # Stage callbacks + runtime model state
        self._hooks = StageHooks()
        self.core_models: list[Any] = []
        self.optional_models: list[Any] = []

        # Config discovery (available before load)
        self._core_specs: list[Any] = []
        self._plugin_interface = plugin_interface

        # State storage
        self.data: Any = None  # For storing InitializationData or similar

    def register_core_models(self, specs: list[Any]) -> None:
        """Register core ModelSpecs for config discovery. No load() needed."""
        self._core_specs = list(specs)

    def _get_all_specs(self) -> list[Any]:
        """Get all ModelSpecs: core + plugin registry."""
        specs = list(self._core_specs)
        seen = {s.name for s in specs}
        if self._plugin_interface is not None:
            from neofoam.core.plugin_system import PluginSystem

            registry = PluginSystem.get_registered(self._plugin_interface.__name__)
            if registry:
                for plugin_cls in registry.plugin_registry:
                    if hasattr(plugin_cls, "get_model_instance"):
                        spec = plugin_cls.get_model_instance(plugin_cls)
                        if spec.name not in seen:
                            seen.add(spec.name)
                            specs.append(spec)
        return specs

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

        # VALIDATE: verify fvSchemes/fvSolution requirements from active operations
        self._verify(load_result)

        if self._hooks.build is None:
            raise RuntimeError(f"No @{self.name}.build defined")
        lazy_inits = self._hooks.build(self.core_models, self.optional_models)

        return execute_initialization(lazy_inits)

    def _verify(self, load_result: LoadResult) -> None:
        """Run verification after RESOLVE. Raises on errors."""
        errors = self._collect_errors(load_result)
        if errors:
            msg_lines = [f"SolverConfigError: {len(errors)} validation error(s)\n"]
            for e in errors:
                msg_lines.append(f"  {e.file_name}: {e.field} — {e.message}")
            raise RuntimeError("\n".join(msg_lines))

    def validate(self) -> list[VerificationError]:
        """Run LOAD + RESOLVE + VERIFY and return errors.

        Does not run BUILD or EXECUTE. Use this to check whether a case
        directory has valid fvSchemes / fvSolution / model configs without
        actually running the solver.

        Returns:
            List of ``VerificationError`` (empty if valid).
        """
        if self._hooks.load is None:
            raise RuntimeError(f"No @{self.name}.load defined")
        load_result = self._hooks.load()
        self.core_models = load_result.core_models
        self.optional_models = load_result.optional_models

        config = ConfigContext()
        for loaded_model in load_result.core_models:
            key = (
                getattr(loaded_model, "name", None)
                or type(loaded_model).__name__.lower()
            )
            if not config.contains(key):
                config.register(key, loaded_model)
        for runtime in load_result.optional_models:
            if not config.contains(runtime.name):
                config.register(runtime.name, runtime)

        if self._hooks.resolve is not None:
            self._hooks.resolve(config)

        return self._collect_errors(load_result)

    def _collect_errors(self, load_result: LoadResult) -> list[VerificationError]:
        """Collect all verification errors without raising."""
        all_models = self.core_models + self.optional_models
        scheme_reqs, solver_reqs = collect_requirements_from_models(all_models)

        errors: list[VerificationError] = []

        if load_result.fv_schemes_config is not None and scheme_reqs:
            fv_schemes = load_result.fv_schemes_config
            data = (
                fv_schemes.model_dump()
                if hasattr(fv_schemes, "model_dump")
                else fv_schemes
            )
            errors.extend(verify_fvschemes(data, scheme_reqs))

        if load_result.fv_solution_config is not None and solver_reqs:
            fv_solution = load_result.fv_solution_config
            data = (
                fv_solution.model_dump()
                if hasattr(fv_solution, "model_dump")
                else fv_solution
            )
            errors.extend(verify_fvsolution(data, solver_reqs))

        config_errors = load_result.validate()
        for ve in config_errors:
            errors.append(
                VerificationError(
                    field=str(ve.field),
                    error_type=ve.error_type,
                    message=ve.message,
                    file_name=ve.file_name,
                    subdict=getattr(ve, "subdict", None),
                    input_value=getattr(ve, "input_value", None),
                )
            )

        return errors

    def solver_inputs(self) -> dict[str, type]:
        """Return all config classes from registered models. No load() needed.

        Discovers config classes from:
        - Core specs (registered via ``register_core_models``)
        - Plugin registry (optional models registered via ``.register_with()``)
        - Loaded models (after ``run()`` has been called)

        Call ``model_json_schema()`` on any returned class for AI introspection.
        """
        result: dict[str, type] = {}
        for spec in self._get_all_specs():
            cls = self._extract_config_class(spec)
            if cls is not None:
                result[spec.name] = cls
        # Also check loaded models (available after run)
        for m in list(self.core_models) + list(self.optional_models):
            name = getattr(m, "name", type(m).__name__)
            if name not in result:
                cls = self._extract_config_class(m)
                if cls is not None:
                    result[name] = cls
        return result

    def scheme_inputs(self) -> type:
        """Build typed Pydantic model from all @fvSchemes.add requirements.

        No load() needed. Each field is typed with the correct scheme union
        (DdtScheme, DivScheme, etc.). Call ``model_json_schema()`` on the
        result for AI introspection.
        """
        from neofoam.foam.verification import (
            build_scheme_model,
            collect_requirements_from_models,
        )

        specs = self._get_all_specs() or (
            list(self.core_models) + list(self.optional_models)
        )
        scheme_reqs, _ = collect_requirements_from_models(specs)
        return build_scheme_model(scheme_reqs)

    def _extract_config_class(self, model: Any) -> Optional[type]:
        """Extract config class from a ModelSpec, ModelRuntime, or similar.

        Tries in order:
        1. ``@spec.config()`` decorator (``_config_class``)
        2. ``@load`` return type annotation
        3. ``runtime.config`` instance type
        4. Scan the module where the spec's operations are defined for BaseConfig subclasses
        """
        import inspect
        from typing import get_type_hints

        config_cls = getattr(model, "_config_class", None)
        if config_cls is not None:
            return config_cls  # type: ignore[return-value]

        load_func = getattr(model, "_load_func", None)
        if load_func is not None:
            try:
                hints = get_type_hints(load_func)
                ret = hints.get("return")
                if (
                    ret is not None
                    and isinstance(ret, type)
                    and issubclass(ret, BaseConfig)
                ):
                    return ret
            except Exception:
                pass

        config = getattr(model, "config", None)
        if config is not None and isinstance(config, BaseConfig):
            return type(config)

        # Fallback: scan the module of the first operation or build func
        funcs = [
            getattr(model, "_build_func", None),
            getattr(model, "_load_func", None),
        ]
        ops = getattr(model, "_operations", [])
        if ops:
            funcs.append(ops[0].func)
        for func in funcs:
            if func is None:
                continue
            module = inspect.getmodule(func)
            if module is None:
                continue
            for attr in vars(module).values():
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseConfig)
                    and attr is not BaseConfig
                ):
                    return attr

        return None

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
