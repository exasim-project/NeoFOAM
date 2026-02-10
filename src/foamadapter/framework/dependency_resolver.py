# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Runtime dependency resolution for Depends() markers.

Handles dependency injection during solver execution with scope management.
"""

import inspect
from typing import Any, Callable, Optional, get_origin, get_args, Annotated

from .context import Context
from .initialization.depends import Depends


class DependencyResolver:
    """
    Resolves Depends() markers at runtime during operation execution.

    Features:
    - Scope-based caching (time_step, iteration, operation)
    - Nested dependency resolution
    - String path resolution ("fields.U", "models.turbulence")
    - Provider function resolution

    Usage:
        resolver = DependencyResolver()

        # Resolve all dependencies for a function
        kwargs = resolver.resolve_arguments(my_func, ctx)
        result = my_func(**kwargs)

        # Clear caches at scope boundaries
        resolver.clear_scope("iteration")  # End of iteration
        resolver.clear_scope("time_step")  # End of time step
    """

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {
            "time_step": {},
            "iteration": {},
            "operation": {},
        }

    def resolve_arguments(
        self, func: Callable, ctx: Optional[Context] = None, **provided_kwargs: Any
    ) -> dict[str, Any]:
        """
        Resolve all Depends() markers and field parameters in function signature.

        Args:
            func: Function to resolve dependencies for
            ctx: Execution context for field/model access
            **provided_kwargs: Already-resolved arguments to pass through

        Returns:
            Dictionary of all arguments to pass to func
        """
        sig = inspect.signature(func)
        kwargs = provided_kwargs.copy()

        for param_name, param in sig.parameters.items():
            if param_name in kwargs:
                continue  # Already provided

            if param_name in ("self", "cls"):
                continue  # Skip instance/class parameters

            # Check for Annotated[Type, Depends(...)]
            depends = self._extract_depends(param.annotation)

            if depends:
                value = self._resolve_dependency(depends, ctx)
                if value is None and not depends.optional:
                    raise ValueError(f"Required dependency '{param_name}' not found")
                kwargs[param_name] = value
            elif param.annotation != inspect.Parameter.empty:
                # Check for Context type
                if param.annotation is Context:
                    kwargs[param_name] = ctx
                    continue

                # Check if it's an Annotated type with string marker
                if get_origin(param.annotation) is Annotated:
                    args = get_args(param.annotation)
                    # Check for Annotated[dict, "models"] or Annotated[dict, "fields"]
                    if len(args) > 1 and isinstance(args[1], str):
                        marker = args[1]
                        if marker == "models" and ctx:
                            # Get from ctx.models
                            kwargs[param_name] = ctx.models.get(param_name)
                        elif marker == "fields" and ctx:
                            # Get from ctx.fields
                            kwargs[param_name] = ctx.fields.get(param_name)
                        else:
                            # Try generic lookup
                            kwargs[param_name] = getattr(ctx, marker, {}).get(
                                param_name
                            )
                else:
                    # Try auto-resolution from context.fields for plain dict parameters
                    if ctx and param_name in ctx.fields:
                        kwargs[param_name] = ctx.fields[param_name]

        return kwargs

    def _extract_depends(self, annotation: Any) -> Optional[Depends]:
        """Extract Depends from Annotated[Type, Depends(...)]."""
        if get_origin(annotation) is Annotated:
            args = get_args(annotation)
            for arg in args[1:]:
                if isinstance(arg, Depends):
                    return arg
        return None

    def _resolve_dependency(self, depends: Depends, ctx: Optional[Context]) -> Any:
        """
        Resolve a single dependency with caching.

        Args:
            depends: Depends marker containing dependency spec
            ctx: Context for field/model access

        Returns:
            Resolved dependency value
        """
        cache_key = str(depends.dependency)
        scope = getattr(depends, "scope", "time_step")
        use_cache = getattr(depends, "cache", True)

        # Check cache
        if use_cache and cache_key in self._cache[scope]:
            return self._cache[scope][cache_key]

        # Resolve based on type
        if isinstance(depends.dependency, str):
            # String path like "fields.U" or "models.turbulence"
            value = self._resolve_path(depends.dependency, ctx)
        elif callable(depends.dependency):
            # Provider function
            value = self._resolve_callable(depends.dependency, ctx)
        else:
            raise ValueError(f"Invalid dependency type: {type(depends.dependency)}")

        # Cache result
        if use_cache:
            self._cache[scope][cache_key] = value

        return value

    def _resolve_path(self, path: str, ctx: Optional[Context]) -> Any:
        """
        Resolve string path like 'fields.U' or 'models.turbulence'.

        Args:
            path: Dot-separated path (e.g., "fields.U", "models.turbulence")
            ctx: Context to resolve from

        Returns:
            Value at path or None if not found
        """
        if ctx is None:
            return None

        parts = path.split(".")

        if parts[0] == "fields":
            return ctx.fields.get(parts[1]) if len(parts) > 1 else None
        elif parts[0] == "models":
            return ctx.models.get(parts[1]) if len(parts) > 1 else None
        else:
            # Try direct attribute access
            return getattr(ctx, path, None)

    def _resolve_callable(self, provider: Callable, ctx: Optional[Context]) -> Any:
        """
        Resolve provider function (may have its own dependencies).

        Args:
            provider: Provider function to call
            ctx: Context for nested dependency resolution

        Returns:
            Result of calling provider with resolved dependencies
        """
        # Recursively resolve provider's dependencies
        kwargs = self.resolve_arguments(provider, ctx)
        return provider(**kwargs)

    def clear_scope(self, scope: str):
        """
        Clear cache for a specific scope.

        Args:
            scope: Scope to clear ("time_step", "iteration", "operation")
        """
        if scope in self._cache:
            self._cache[scope].clear()

    def clear_all(self):
        """Clear all caches."""
        for scope in self._cache:
            self._cache[scope].clear()
