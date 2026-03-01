# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Unified operation wrapping: config injection + dependency resolution.

Single module replacing config_injection.py and dependency_resolver.py.
Provides one public entry point — ``wrap_operation()`` — that handles
both BaseConfig injection and Depends() resolution.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Annotated, Any, Callable, Optional, get_args, get_origin

from neofoam.io import BaseConfig

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization.depends import Depends


# ---------------------------------------------------------------------------
# Config discovery helpers
# ---------------------------------------------------------------------------


def discover_configs_from_signature(func: Callable[..., Any]) -> list[dict[str, Any]]:
    """Return [{param_name, config_type}] for BaseConfig parameters in *func*."""
    sig = inspect.signature(func)
    discovered: list[dict[str, Any]] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "ctx"):
            continue
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            continue
        if getattr(annotation, "__origin__", None) is not None:
            continue
        try:
            if isinstance(annotation, type) and issubclass(annotation, BaseConfig):
                discovered.append({"param_name": param_name, "config_type": annotation})
        except TypeError:
            continue

    return discovered


def discover_call_metadata(func: Callable[..., Any]) -> dict[str, Any]:
    """Pre-compute metadata for inject_and_call: first_param_name, has_ctx, config_params."""
    sig = inspect.signature(func)
    params = list(sig.parameters)
    return {
        "first_param_name": params[0],
        "has_ctx": "ctx" in sig.parameters,
        "config_params": discover_configs_from_signature(func),
    }


def inject_and_call(
    func: Callable[..., Any],
    runtime: Any,
    call_meta: dict[str, Any],
    ctx: Any = None,
) -> Any:
    """Call *func* with self=runtime, optional ctx, and injected BaseConfig params.

    *call_meta* is a pre-computed dict from ``discover_call_metadata`` containing
    first_param_name, has_ctx, and config_params.

    The first parameter is always bound to *runtime* regardless of its name
    (commonly ``self`` or ``_self``).
    """
    kwargs: dict[str, Any] = {call_meta["first_param_name"]: runtime}

    if call_meta["has_ctx"] and ctx is not None:
        kwargs["ctx"] = ctx

    for cfg_meta in call_meta["config_params"]:
        kwargs[cfg_meta["param_name"]] = find_config_by_type(
            runtime.config, cfg_meta["config_type"]
        )

    return func(**kwargs)


def find_config_by_type(runtime_config: Any, config_type: type) -> Any:
    """
    Find a config instance of *config_type* in *runtime_config*.

    Handles two cases:
    - Direct match: ``runtime_config`` is already an instance of ``config_type``.
    - Namespace: ``runtime_config`` is a SimpleNamespace whose attributes
      include an instance of ``config_type``.
    """
    if isinstance(runtime_config, config_type):
        return runtime_config
    if hasattr(runtime_config, "__dict__"):
        for val in vars(runtime_config).values():
            if isinstance(val, config_type):
                return val
    raise ValueError(
        f"No config of type {config_type.__name__} found in runtime.config "
        f"(type: {type(runtime_config).__name__})"
    )


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


class DependencyResolver:
    """Resolve Depends() markers and context-backed arguments."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {
            "time_step": {},
            "iteration": {},
            "operation": {},
        }

    def resolve_arguments(
        self,
        func: Callable[..., Any],
        ctx: Optional[Context] = None,
        **provided_kwargs: Any,
    ) -> dict[str, Any]:
        """Resolve function arguments from Depends markers and Context."""
        sig = inspect.signature(func)
        kwargs = provided_kwargs.copy()

        for param_name, param in sig.parameters.items():
            if param_name in kwargs:
                continue
            if param_name in ("self", "cls"):
                continue

            depends = self._extract_depends(param.annotation)
            if depends:
                value = self._resolve_dependency(depends, ctx)
                if value is None and not depends.optional:
                    raise ValueError(f"Required dependency '{param_name}' not found")
                kwargs[param_name] = value
                continue

            if param.annotation != inspect.Parameter.empty:
                if param.annotation is Context:
                    kwargs[param_name] = ctx
                    continue

                if get_origin(param.annotation) is Annotated:
                    args = get_args(param.annotation)
                    if len(args) > 1 and isinstance(args[1], str):
                        marker = args[1]
                        if marker == "models" and ctx:
                            kwargs[param_name] = ctx.models.get(param_name)
                            continue
                        if marker == "fields" and ctx:
                            kwargs[param_name] = ctx.fields.get(param_name)
                            continue
                        if ctx:
                            kwargs[param_name] = getattr(ctx, marker, {}).get(
                                param_name
                            )
                            continue

            if ctx and param_name in ctx.fields:
                kwargs[param_name] = ctx.fields[param_name]

        return kwargs

    def _extract_depends(self, annotation: Any) -> Optional[Depends]:
        if get_origin(annotation) is Annotated:
            for arg in get_args(annotation)[1:]:
                if isinstance(arg, Depends):
                    return arg
        return None

    def _resolve_dependency(self, depends: Depends, ctx: Optional[Context]) -> Any:
        cache_key = str(depends.dependency)
        scope = getattr(depends, "scope", "time_step")
        use_cache = getattr(depends, "cache", True)

        if use_cache and cache_key in self._cache[scope]:
            return self._cache[scope][cache_key]

        if isinstance(depends.dependency, str):
            value = self._resolve_path(depends.dependency, ctx)
        elif callable(depends.dependency):
            value = self._resolve_callable(depends.dependency, ctx)
        else:
            raise ValueError(f"Invalid dependency type: {type(depends.dependency)}")

        if use_cache:
            self._cache[scope][cache_key] = value

        return value

    def _resolve_path(self, path: str, ctx: Optional[Context]) -> Any:
        if ctx is None:
            return None

        parts = path.split(".")
        if parts[0] == "fields":
            return ctx.fields.get(parts[1]) if len(parts) > 1 else None
        if parts[0] == "models":
            return ctx.models.get(parts[1]) if len(parts) > 1 else None
        return getattr(ctx, path, None)

    def _resolve_callable(
        self, provider: Callable[..., Any], ctx: Optional[Context]
    ) -> Any:
        kwargs = self.resolve_arguments(provider, ctx)
        return provider(**kwargs)

    def clear_scope(self, scope: str) -> None:
        if scope in self._cache:
            self._cache[scope].clear()

    def clear_all(self) -> None:
        for scope in self._cache:
            self._cache[scope].clear()


# ---------------------------------------------------------------------------
# Unified wrap_operation
# ---------------------------------------------------------------------------


def wrap_operation(
    func: Callable[..., Any],
    runtime: Any,
    dependency_resolver: DependencyResolver,
) -> Callable[[Context], Any]:
    """Wrap *func* so it can be called with just a Context.

    This is the single public entry point replacing the if/else branch
    that previously existed in both ModelSpec and SolverSpec.

    Strategy:
    - If *func* has BaseConfig-annotated parameters, use config injection
      (resolve configs from ``runtime.config``).
    - Otherwise, fall through to dependency resolution via
      *dependency_resolver*.

    In both cases, ``self`` is bound to *runtime* and ``FieldUpdates``
    return values are applied to the context automatically.
    """
    discovered = discover_configs_from_signature(func)

    if discovered:
        return _create_config_wrapper(func, discovered, runtime)
    else:
        return _create_dependency_wrapper(func, runtime, dependency_resolver)


def _create_config_wrapper(
    func: Callable[..., Any],
    discovered: list[dict[str, Any]],
    runtime: Any,
) -> Callable[[Context], Any]:
    """Wrap an operation function to inject config from ``runtime.config``."""
    sig = inspect.signature(func)
    expects_self = "self" in sig.parameters
    expects_ctx = "ctx" in sig.parameters

    @wraps(func)
    def wrapper(ctx: Context) -> Any:
        call_kwargs: dict[str, Any] = {}

        if expects_ctx:
            call_kwargs["ctx"] = ctx
        if expects_self:
            call_kwargs["self"] = runtime

        for cfg_meta in discovered:
            param_name = cfg_meta["param_name"]
            config_type = cfg_meta["config_type"]
            call_kwargs[param_name] = find_config_by_type(runtime.config, config_type)

        for pname, param in sig.parameters.items():
            if pname in ("self", "ctx") or pname in call_kwargs:
                continue
            if (
                param.annotation in (float, int, str, bool)
                or param.annotation is inspect.Parameter.empty
            ):
                if hasattr(ctx, "fields") and pname in ctx.fields:
                    call_kwargs[pname] = ctx.fields[pname]

        result = func(**call_kwargs)

        if isinstance(result, FieldUpdates):
            ctx.fields.update(result)
            return None

        return result

    setattr(wrapper, "_already_wrapped", True)
    return wrapper


def _create_dependency_wrapper(
    func: Callable[..., Any],
    instance: Any,
    dependency_resolver: DependencyResolver,
) -> Callable[[Context], Any]:
    """Wrap *func* using dependency resolution for argument injection."""

    @wraps(func)
    def wrapper(ctx: Context) -> Any:
        kwargs = dependency_resolver.resolve_arguments(func, ctx)

        sig = inspect.signature(func)
        if "self" in sig.parameters and "self" not in kwargs:
            kwargs["self"] = instance

        result = func(**kwargs)

        if isinstance(result, FieldUpdates):
            ctx.fields.update(result)
            return None

        return result

    return wrapper
