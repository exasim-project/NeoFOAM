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

                # Model-owned interface: a param annotated with a ModelInterface
                # handle is injected as the owning runtime's per-case bound handle.
                from .model.interface import ModelInterface as _ModelInterface  # noqa: PLC0415
                from .model.runtime import ModelRuntime as _ModelRuntime  # noqa: PLC0415

                if isinstance(param.annotation, _ModelInterface):
                    iface = param.annotation
                    if ctx is None:
                        raise ValueError(
                            f"Parameter '{param_name}' is typed as a ModelInterface "
                            "but no Context was provided to the resolver."
                        )
                    owner_runtime = ctx.models.get(iface.owner.name)
                    if (
                        not isinstance(owner_runtime, _ModelRuntime)
                        or iface.name not in owner_runtime.bound_interfaces
                    ):
                        raise ValueError(
                            f"Interface '{iface.name}' (owned by model "
                            f"'{iface.owner.name}') is not bound for this case; "
                            "expected a BoundModelInterface on the owning "
                            "ModelRuntime."
                        )
                    kwargs[param_name] = owner_runtime.bound_interfaces[iface.name]
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


def wrap_with_dependency_resolution(
    func: Callable[..., Any],
    instance: Any,
    dependency_resolver: DependencyResolver,
) -> Callable[[Context], Any]:
    """Wrap *func* so it can be called with just a Context.

    Dependencies are resolved via *dependency_resolver*.  If the function
    signature includes a ``self`` parameter it is bound to *instance*.
    If the return value is a ``FieldUpdates`` the context is updated
    automatically.

    This is the canonical implementation shared by ``SolverSpec`` and
    ``ModelSpec`` — avoids duplicating the same wrapper in every factory.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(ctx: Context) -> Any:
        kwargs = dependency_resolver.resolve_arguments(func, ctx)

        sig = inspect.signature(func)
        if "self" in sig.parameters and "self" not in kwargs:
            kwargs["self"] = instance

        result = func(**kwargs)

        from .context import FieldUpdates

        if isinstance(result, FieldUpdates):
            ctx.fields.update(result)
            return None

        return result

    return wrapper
