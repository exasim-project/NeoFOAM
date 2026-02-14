# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Runtime dependency resolution for Depends() markers.

Handles dependency injection during solver/model execution.
"""

import inspect
from typing import Annotated, Any, Callable, Optional, get_args, get_origin

from .context import Context
from .initialization.depends import Depends


class DependencyResolver:
    """Resolve Depends() markers and context-backed arguments."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {
            "time_step": {},
            "iteration": {},
            "operation": {},
        }

    def resolve_arguments(
        self, func: Callable, ctx: Optional[Context] = None, **provided_kwargs: Any
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
                            kwargs[param_name] = getattr(ctx, marker, {}).get(param_name)
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

    def _resolve_callable(self, provider: Callable, ctx: Optional[Context]) -> Any:
        kwargs = self.resolve_arguments(provider, ctx)
        return provider(**kwargs)

    def clear_scope(self, scope: str) -> None:
        if scope in self._cache:
            self._cache[scope].clear()

    def clear_all(self) -> None:
        for scope in self._cache:
            self._cache[scope].clear()
