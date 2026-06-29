# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""ModelInterface — an extension point owned by a Model.

A model declares an interface it owns via ``@<model>.interface`` decorating the
fold function: the decorated name is the interface name and the function is the
single fold (defining the empty case). The decorator returns a ``ModelInterface``
handle, bound to the name it decorates and registered on its owning model, so the
handle doubles as a ``@<model>.contributes(<interface>)`` target and as an
operation-parameter annotation.

A ``BoundModelInterface`` is the per-case, live counterpart: it holds the active
contributing model runtimes gathered for one case plus the live ``Context`` and,
when called with no arguments, folds the active contributions. It lives on the
owning model's ``ModelRuntime`` (``runtime.bound_interfaces``) — never on the
Context, and there is no process-global registry here.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterable, TypeVar

if TYPE_CHECKING:
    from .runtime import ModelRuntime
    from .spec import ModelSpec

T = TypeVar("T")


class ModelInterface(Generic[T]):
    """An extension point owned by a model: a fold plus its contributions."""

    def __init__(
        self,
        name: str,
        owner: ModelSpec,
        fold: Callable[[Iterable[T]], T],
    ) -> None:
        self.name = name
        self.owner = owner
        self._fold = fold
        # One insertion-ordered relation: contribution function -> owning model.
        self._contributions: dict[Callable[..., T], ModelSpec] = {}

    @property
    def contributions(self) -> tuple[Callable[..., T], ...]:
        """The registered contribution functions, in registration order."""
        return tuple(self._contributions)

    def owner_of(self, contribution: Callable[..., T]) -> ModelSpec:
        """Return the contributing model that registered *contribution*.

        Raises:
            KeyError: if *contribution* was never registered on this interface.
        """
        if contribution not in self._contributions:
            raise KeyError(
                f"interface '{self.name}': {contribution!r} is not a registered "
                "contribution."
            )
        return self._contributions[contribution]

    def fold(self, values: Iterable[T]) -> T:
        """Combine *values* via the declared fold (empty -> the fold's default)."""
        return self._fold(values)

    def _register_contribution(
        self, func: Callable[..., T], owner: ModelSpec
    ) -> Callable[..., T]:
        """Record *func* as a contribution owned by the *owner* model."""
        self._contributions[func] = owner
        return func


def _resolve_contribution_kwargs(
    interface_name: str,
    func: Callable[..., T],
    runtime: ModelRuntime,
    ctx: Any,
) -> dict[str, Any]:
    """Resolve *func*'s parameters the same way ``@model.operation`` does.

    Config-typed params (``BaseConfig`` subclasses) are pulled from the
    *contributing* runtime's own ``config`` by type via ``config_injection``; the
    remainder is resolved by the shared ``DependencyResolver`` (``ctx.fields`` by
    name, ``Depends`` markers). A parameter that resolves to neither raises a
    ``ValueError`` naming the interface, the contribution, and the parameter.
    """
    # Lazy imports break the cycle interface -> config_injection/dependency_resolver
    # -> (resolver) -> model.interface.
    from neofoam.framework.config_injection import (  # noqa: PLC0415
        _discover_configs_from_signature,
        _find_config_by_type,
    )
    from neofoam.framework.dependency_resolver import (  # noqa: PLC0415
        DependencyResolver,
    )

    preresolved: dict[str, Any] = {}
    for meta in _discover_configs_from_signature(func):
        pname = meta["param_name"]
        try:
            preresolved[pname] = _find_config_by_type(
                runtime.config, meta["config_type"]
            )
        except ValueError as exc:
            raise ValueError(
                f"interface '{interface_name}': contribution '{func.__name__}' "
                f"(model '{runtime.spec.name}') could not resolve config "
                f"parameter '{pname}': {exc}"
            ) from exc

    resolved = DependencyResolver().resolve_arguments(func, ctx=ctx, **preresolved)

    sig = inspect.signature(func)
    kwargs: dict[str, Any] = {}
    for pname in sig.parameters:
        if pname in ("self", "cls"):
            kwargs[pname] = None  # operation-method form: skip the receiver
            continue
        if pname not in resolved:
            available = sorted(ctx.fields) if ctx is not None else []
            raise ValueError(
                f"interface '{interface_name}': contribution '{func.__name__}' "
                f"(model '{runtime.spec.name}') declares parameter '{pname}' but no "
                f"provider supplies it. Available context fields: {available}"
            )
        kwargs[pname] = resolved[pname]
    return kwargs


class BoundModelInterface(Generic[T]):
    """An interface bound to one case: active contributing runtimes + a Context.

    Built per case on the owning model's runtime (``runtime.bound_interfaces``).
    Calling it with no arguments folds exactly the contributions whose contributing
    model is active for this case (i.e. has a runtime in *contributing_runtimes*),
    resolving each contribution's params from that contributing runtime's own
    config + the bound Context's fields.
    """

    def __init__(
        self,
        interface: ModelInterface[T],
        contributing_runtimes: Iterable[ModelRuntime],
        ctx: Any,
    ) -> None:
        self._interface = interface
        self._runtimes = list(contributing_runtimes)
        self._ctx = ctx
        self._runtime_by_spec = {rt.spec: rt for rt in self._runtimes}

    def __call__(self, ctx: Any = None) -> T:
        """Fold the active contributions against the **call-time** Context.

        *ctx* (the live Context for this step) overrides the bind-time Context so
        the owner runtime can be wired with an empty Context (no mesh-bound pybFoam
        capture -> no cross-run GC cycle) and still fold against live fields/models.
        With no *ctx* the bind-time Context is used (the original call form).
        """
        live_ctx = ctx if ctx is not None else self._ctx
        values: list[T] = []
        for func in self._interface.contributions:
            owner_spec = self._interface.owner_of(func)
            runtime = self._runtime_by_spec.get(owner_spec)
            if runtime is None:
                continue  # contributing model not active for this case
            kwargs = _resolve_contribution_kwargs(
                self._interface.name, func, runtime, live_ctx
            )
            values.append(func(**kwargs))
        return self._interface.fold(values)


def bind_model_interface(
    owner_runtime: ModelRuntime,
    interface: ModelInterface[T],
    contributing_runtimes: Iterable[ModelRuntime],
    ctx: Any,
) -> BoundModelInterface[T]:
    """Build a :class:`BoundModelInterface` and store it on *owner_runtime*.

    This is the per-case gather seam: given the case's active contributing
    runtimes, it binds them (plus the live Context) onto the owning runtime under
    ``owner_runtime.bound_interfaces[interface.name]`` and returns the bound handle.
    The ``@<owner>.build`` `InitStep` that calls this automatically lands in iter-3;
    iter-2 exercises it directly.
    """
    bound = BoundModelInterface(interface, contributing_runtimes, ctx)
    owner_runtime.bound_interfaces[interface.name] = bound
    return bound


def active_contributors(
    interface: ModelInterface[T],
    candidate_runtimes: Iterable[ModelRuntime],
) -> list[ModelRuntime]:
    """The subset of *candidate_runtimes* whose spec contributes to *interface*.

    Matched by ``ModelSpec`` **identity** (not name), so a contributor in any
    family folds into an interface owned by any model.
    """
    contributing = {interface.owner_of(f) for f in interface.contributions}
    return [rt for rt in candidate_runtimes if rt.spec in contributing]


def bind_owned_interfaces(
    owner_runtime: ModelRuntime,
    candidate_runtimes: Iterable[ModelRuntime],
    ctx: Any,
) -> ModelRuntime:
    """Bind every interface *owner_runtime*'s spec declares to the active
    contributors among *candidate_runtimes*, storing each
    :class:`BoundModelInterface` on ``owner_runtime.bound_interfaces``.

    This is the per-case ``@build`` auto-wiring seam: it returns *owner_runtime*
    so the caller can register it in ``ctx.models`` under ``owner.name`` (the key
    the resolver looks up).
    """
    candidates = list(candidate_runtimes)
    for iface in owner_runtime.spec.declared_interfaces.values():
        bind_model_interface(
            owner_runtime, iface, active_contributors(iface, candidates), ctx
        )
    return owner_runtime
