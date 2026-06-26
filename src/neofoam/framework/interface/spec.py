# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
InterfaceSpec — the fold-kernel for a named gather point.

``Interface(name)`` mirrors ``Model(name)`` from :mod:`neofoam.framework.model`:
it returns an ``InterfaceSpec`` that owns a ``@combine`` fold function and a
``@contribute`` registration decorator.

Contributions are gated by **ownership**: ``@<iface>.contribute(model=<spec>)``
binds a contribution to a model; :meth:`collect` folds it only when that model's
name is a key in ``ctx.models``. A contribution registered without a ``model=``
is unowned and always folds. There is no mutable active-set — gating is a pure
function of the registered contributions and the live Context.

Convention note: ``Interface`` is a module-level factory function with a
capital letter to match the ``Model(name)`` pattern — it is not a class.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar, overload

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
    Calling it returns ``spec.collect(ctx)``.
    """

    def __init__(self, spec: "InterfaceSpec[T]", ctx: Any) -> None:
        self._spec = spec
        self._ctx = ctx

    def __call__(self) -> T:
        """Return ``spec.collect(self._ctx)``.

        The fold runs over every contribution whose owning model is active in the
        bound Context (its ``name`` is a key in ``ctx.models``); unowned
        contributions always fold.
        """
        return self._spec.collect(self._ctx)


class InterfaceSpec(Generic[T]):
    """
    Immutable definition of a named gather point.

    Holds exactly one ``@combine`` fold function (registered once at module
    import), a list of ``@contribute`` contribution functions, and an owner map
    binding each contribution to the model that gates it (``None`` for an
    unowned, always-active contribution).

    ``_collect_contributions`` accepts a ``providers: dict[str, Any]`` mapping
    parameter names to values and delegates resolution to
    :class:`~neofoam.framework.dependency_resolver.DependencyResolver` — the
    same resolver used by ``@model.operation``.  It is the Context-free path and
    folds **every** registered contribution (it has no active-model set to read).

    :meth:`collect` is the Context-backed path: it folds a contribution only when
    its owning model's name is a key in ``ctx.models`` (unowned contributions
    always fold).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._combine_func: Optional[Callable[[Iterable[T]], T]] = None
        self._contributions: list[Callable[..., T]] = []
        self._owner: dict[Callable[..., T], Any] = {}

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

    @overload
    def contribute(
        self, func: Callable[..., T], *, model: Any = None
    ) -> Callable[..., T]: ...

    @overload
    def contribute(
        self, func: None = None, *, model: Any = None
    ) -> Callable[[Callable[..., T]], Callable[..., T]]: ...

    def contribute(
        self,
        func: Optional[Callable[..., T]] = None,
        *,
        model: Any = None,
    ) -> Any:
        """Register a contribution (operation-style; deps inferred from params).

        Usage::

            @iface.contribute                    # unowned -> always folds
            @iface.contribute(model=<ModelSpec>) # gated by that model

        An owned contribution folds in :meth:`collect` only when its owning
        model's ``name`` is a key in ``ctx.models``; an unowned one always folds.
        The contribution must not declare a parameter typed ``Context``.

        Raises:
            ValueError: if *func* declares a parameter typed ``Context``.
        """
        if func is None:

            def decorator(f: Callable[..., T]) -> Callable[..., T]:
                return self._register_contribution(f, model)

            return decorator
        return self._register_contribution(func, model)

    def _register_contribution(
        self, func: Callable[..., T], model: Any
    ) -> Callable[..., T]:
        from neofoam.framework.context import Context as _Context

        if model is not None and not hasattr(model, "name"):
            raise TypeError(
                f"Interface '{self.name}': contribution '{func.__name__}' was given "
                f"model={model!r}, which has no '.name' attribute. The owning model "
                "must be a ModelSpec (or expose '.name') so collect() can gate it "
                "against ctx.models."
            )

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
        self._owner[func] = model
        return func

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
        """Resolve every registered contribution against *providers* and fold.

        Intentionally **ungated**: it ignores ownership and folds *all* registered
        contributions. It exists only for Context-free unit testing of the
        resolve+fold mechanism; :meth:`collect` is the gated production path that
        respects model ownership. Do not call this from solver code.

        This is the Context-free path (no active-model set to read); it folds
        all registered contributions.  Each function's parameters are resolved by
        :class:`~neofoam.framework.dependency_resolver.DependencyResolver`
        (the same resolver used by ``@model.operation``):

        - ``Depends(callable)`` markers are called at resolution time.
        - Plain unannotated parameters are matched by name against *providers*.
        - ``"self"`` / ``"cls"`` are skipped automatically by the resolver.

        If any plain parameter remains unresolved after the resolver runs, a
        ``ValueError`` is raised immediately — never a silent skip.

        Raises:
            ValueError: if a contribution parameter cannot be resolved.
            RuntimeError: if no ``@combine`` function has been registered.
        """
        resolver = DependencyResolver()
        bound: list[Callable[[], T]] = []
        for fn in self._contributions:
            kwargs = resolver.resolve_arguments(fn, ctx=None, **providers)
            sig = inspect.signature(fn)
            fn_params = set(sig.parameters.keys()) - {"self", "cls"}
            for param_name in fn_params:
                if param_name not in kwargs:
                    raise ValueError(
                        f"Interface '{self.name}': contribution "
                        f"'{fn.__name__}' declares parameter '{param_name}' "
                        f"but no provider supplies it. "
                        f"Available providers: {sorted(providers.keys())}"
                    )
            fn_kwargs = {k: kwargs[k] for k in fn_params if k in kwargs}
            for skipped in ("self", "cls"):
                if skipped in sig.parameters and skipped not in fn_kwargs:
                    fn_kwargs[skipped] = None
            bound.append(_make_caller(fn, dict(fn_kwargs)))
        return self._collect_values(bound)

    def collect(self, ctx: Any) -> T:
        """Fold the contributions whose owning model is active in *ctx*.

        A contribution folds iff it is unowned (``self._owner[fn] is None``) or
        its owning model's ``name`` is a key in ``ctx.models``.  Parameters are
        resolved via the same
        :class:`~neofoam.framework.dependency_resolver.DependencyResolver`
        path as ``@model.operation``.  No activation state is stored — two
        Contexts folded in one process do not leak into each other.

        Args:
            ctx: A live :class:`~neofoam.framework.context.Context` instance.

        Raises:
            ValueError: if a folded contribution's parameter cannot be resolved.
            RuntimeError: if no ``@combine`` function has been registered.
        """
        active_models = ctx.models if ctx is not None else {}
        resolver = DependencyResolver()
        bound: list[Callable[[], T]] = []
        for fn in self._contributions:
            owner = self._owner.get(fn)
            if owner is not None and owner.name not in active_models:
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
