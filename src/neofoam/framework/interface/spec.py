# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
InterfaceSpec — the fold-kernel for a named gather point.

``Interface(name)`` mirrors ``Model(name)`` from :mod:`neofoam.framework.model`:
it returns an ``InterfaceSpec`` that owns a ``@combine`` fold function and a
``@contribute`` registration decorator.

Convention note: ``Interface`` is a module-level factory function with a
capital letter to match the ``Model(name)`` pattern — it is not a class.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

from neofoam.framework.dependency_resolver import DependencyResolver

T = TypeVar("T")


def _make_caller(f: Callable[..., T], kw: dict[str, Any]) -> Callable[[], T]:
    """Return a zero-argument callable that invokes *f* with keyword args *kw*."""

    def _call() -> T:
        return f(**kw)

    return _call


class BoundInterface(Generic[T]):
    """An interface bound to a live Context, callable to get the fold.

    Produced by :class:`~neofoam.framework.dependency_resolver.DependencyResolver`
    when it encounters a parameter annotated with an :class:`InterfaceSpec` instance.
    Calling it returns ``spec.collect(ctx)`` over the currently active contributions.
    """

    def __init__(self, spec: "InterfaceSpec[T]", ctx: Any) -> None:
        self._spec = spec
        self._ctx = ctx

    def __call__(self) -> T:
        return self._spec.collect(self._ctx)


class InterfaceSpec(Generic[T]):
    """
    Immutable definition of a named gather point.

    Holds exactly one ``@combine`` fold function (registered once at module
    import), a list of ``@contribute`` contribution functions, and exposes
    ``_collect_values`` (fold over zero-argument callables) and
    ``_collect_contributions`` (fold over DI-resolved contribution functions).

    ``_collect_contributions`` accepts a ``providers: dict[str, Any]`` mapping
    parameter names to values and delegates resolution to
    :class:`~neofoam.framework.dependency_resolver.DependencyResolver` — the
    same resolver used by ``@model.operation``.  ``Depends``-annotated params
    are resolved via the resolver; plain params are matched by name from
    *providers*.  If any plain parameter has no matching entry a ``ValueError``
    is raised immediately (never a silent skip).

    ``_collect_values`` is a lower-level helper retained for tests and for
    callers that already have zero-argument callables.

    All registered contributions are active by default.  Use :meth:`deactivate`
    to exclude a contribution from folds without unregistering it, and
    :meth:`activate` to re-enable it.  :meth:`collect` folds only the active
    contributions using the full Context-backed resolver path.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._combine_func: Optional[Callable[[Iterable[T]], T]] = None
        self._contributions: list[Callable[..., T]] = []
        self._active_contributions: set[Callable[..., T]] = set()

    def combine(self, func: Callable[[Iterable[T]], T]) -> Callable[[Iterable[T]], T]:
        """Register the aggregation fold (single registration only).

        Raises:
            RuntimeError: if a ``@combine`` function is already registered.
        """
        if self._combine_func is not None:
            raise RuntimeError(
                f"Interface '{self.name}': a @combine function is already registered. "
                "Only one @combine is allowed per InterfaceSpec."
            )
        self._combine_func = func
        return func

    def contribute(self, func: Callable[..., T]) -> Callable[..., T]:
        """Register a contribution function (operation-style; deps inferred from params).

        The contribution must not declare a parameter typed ``Context``.  Its
        parameters are resolved at fold time via
        :class:`~neofoam.framework.dependency_resolver.DependencyResolver`,
        using the same annotation-based injection path as ``@model.operation``.

        The contribution is added to ``_active_contributions`` by default.

        Raises:
            ValueError: if *func* declares a parameter typed ``Context``.
        """
        from neofoam.framework.context import Context as _Context

        # Use get_type_hints to handle `from __future__ import annotations`
        # (which stores annotations as strings rather than live types).
        try:
            hints = typing.get_type_hints(func)
        except Exception:
            hints = {}
        sig = inspect.signature(func)
        for param_name in sig.parameters:
            if param_name in ("self", "cls"):
                continue
            if hints.get(param_name) is _Context:
                raise ValueError(
                    f"Interface '{self.name}': contribution '{func.__name__}' "
                    f"declares parameter '{param_name}: Context'. "
                    "Contributions must not receive a Context object."
                )
        self._contributions.append(func)
        self._active_contributions.add(func)
        return func

    def activate(self, func: Callable[..., T]) -> None:
        """Mark a registered contribution as active (included in folds).

        Raises:
            ValueError: if *func* is not registered on this spec.
        """
        if func not in self._contributions:
            raise ValueError(
                f"Interface '{self.name}': cannot activate unregistered function "
                f"'{func.__name__}'"
            )
        self._active_contributions.add(func)

    def deactivate(self, func: Callable[..., T]) -> None:
        """Mark a registered contribution as inactive (excluded from folds).

        No-op if *func* is already inactive or not registered.
        """
        self._active_contributions.discard(func)

    def _collect_values(self, providers: Iterable[Callable[[], T]]) -> T:
        """Call the registered ``@combine`` fold over values from *providers*.

        Each provider is a zero-argument callable that returns one ``T`` value.
        The results are passed lazily to ``@combine`` as an ``Iterable[T]``.

        Raises:
            RuntimeError: if no ``@combine`` has been registered.
        """
        if self._combine_func is None:
            raise RuntimeError(
                f"Interface '{self.name}': no @combine function registered. "
                "Decorate one function with @<iface>.combine before calling "
                "_collect_values."
            )
        return self._combine_func(p() for p in providers)

    def _collect_contributions(self, providers: dict[str, Any]) -> T:
        """Resolve each active contribution against *providers* and fold.

        Each contribution function's parameters are resolved by
        :class:`~neofoam.framework.dependency_resolver.DependencyResolver`
        (the same resolver used by ``@model.operation``):

        - ``Depends(callable)`` markers are called at resolution time.
        - Plain unannotated parameters are matched by name against *providers*.
        - ``"self"`` / ``"cls"`` are skipped automatically by the resolver.
        - Contributions not in ``_active_contributions`` are skipped.

        If any plain parameter remains unresolved after the resolver runs, a
        ``ValueError`` is raised immediately — never a silent skip (IF12).

        Raises:
            ValueError: if a contribution parameter cannot be resolved.
            RuntimeError: if no ``@combine`` function has been registered.
        """
        resolver = DependencyResolver()
        bound: list[Callable[[], T]] = []
        for fn in self._contributions:
            if fn not in self._active_contributions:
                continue
            kwargs = resolver.resolve_arguments(fn, ctx=None, **providers)
            sig = inspect.signature(fn)
            fn_params = set(sig.parameters.keys()) - {"self", "cls"}
            # Post-check: plain params not handled by the resolver (no Depends
            # marker, not in providers) must raise explicitly — never a silent
            # skip.
            for param_name in fn_params:
                if param_name not in kwargs:
                    raise ValueError(
                        f"Interface '{self.name}': contribution "
                        f"'{fn.__name__}' declares parameter '{param_name}' "
                        f"but no provider supplies it. "
                        f"Available providers: {sorted(providers.keys())}"
                    )
            # Pass only the params the function actually declares, not all
            # providers (extras would cause a TypeError).
            fn_kwargs = {k: kwargs[k] for k in fn_params if k in kwargs}
            # An unbound contribution may still declare a leading "self"/"cls"
            # parameter; it is not a provider but must be supplied to call the
            # function. Pass None — the value is ignored by such contributions.
            for skipped in ("self", "cls"):
                if skipped in sig.parameters and skipped not in fn_kwargs:
                    fn_kwargs[skipped] = None
            bound.append(_make_caller(fn, dict(fn_kwargs)))
        return self._collect_values(bound)

    def collect(self, ctx: Any) -> T:
        """Fold the currently active contributions, resolving params via *ctx*.

        Uses the same :class:`~neofoam.framework.dependency_resolver.DependencyResolver`
        path as ``@model.operation``.  Only contributions that are active
        (all by default; see :meth:`deactivate`) are included in the fold.

        The lazy import of :class:`~neofoam.framework.dependency_resolver.DependencyResolver`
        is already at module top-level; ``Context`` is imported lazily inside
        :meth:`contribute` to keep the circular-import chain clean.

        Args:
            ctx: A live :class:`~neofoam.framework.context.Context` instance.

        Raises:
            ValueError: if a contribution parameter cannot be resolved from *ctx*.
            RuntimeError: if no ``@combine`` function has been registered.
        """
        resolver = DependencyResolver()
        bound: list[Callable[[], T]] = []
        for fn in self._contributions:
            if fn not in self._active_contributions:
                continue
            kwargs = resolver.resolve_arguments(fn, ctx=ctx)
            sig = inspect.signature(fn)
            fn_params = set(sig.parameters.keys()) - {"self", "cls"}
            for param_name in fn_params:
                if param_name not in kwargs:
                    available = sorted(ctx.fields.keys()) if ctx is not None else []
                    raise ValueError(
                        f"Interface '{self.name}': contribution "
                        f"'{fn.__name__}' declares parameter '{param_name}' "
                        f"but no provider supplies it. "
                        f"Available context fields: {available}"
                    )
            fn_kwargs = {k: kwargs[k] for k in fn_params if k in kwargs}
            for skipped in ("self", "cls"):
                if skipped in sig.parameters and skipped not in fn_kwargs:
                    fn_kwargs[skipped] = None
            bound.append(_make_caller(fn, dict(fn_kwargs)))
        return self._collect_values(bound)


def Interface(name: str) -> InterfaceSpec[Any]:
    """Factory: create a named InterfaceSpec.

    Mirrors :func:`neofoam.framework.model.Model` — ``Interface(name)``
    returns an ``InterfaceSpec``.  The capital-letter name is an intentional
    convention match with ``Model(name)``.
    """
    return InterfaceSpec(name)
