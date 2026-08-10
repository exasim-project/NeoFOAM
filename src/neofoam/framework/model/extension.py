# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Extension — the bundle of hooks an operation module declares for models to extend.

An operation module declares one :class:`Extension` per extensible operation and
one ``@<extension>.defines`` function per :class:`Hook`; models register plain
functions with ``@<model>.contributes(<hook>)``.

A contribution runs iff its model is active for the case (a live ``ModelRuntime``
in ``ctx.models``), so the operations never name a model and a case activates
contributions by its files alone.
"""

from __future__ import annotations

import inspect
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterable, overload

if TYPE_CHECKING:
    from .spec import ModelSpec


class Kind(Enum):
    """How a hook combines what its contributions return.

    Declared once per hook at ``@<extension>.defines(kind=...)``, so a
    declaration states which of the three shapes it is instead of inventing a
    combine rule of its own::

        @pressure_extension.defines(kind=Kind.PIPELINE)
        def predicted_flux(phiHbyA: Any) -> Any: ...   # body is documentation

    ``ADDITIVE`` (the default) runs every contribution on the same arguments and
    hands the results to the declaration body, which folds them. ``PIPELINE``
    threads the value instead: each contribution receives the previous one's
    output in the declaration's *first* parameter, the call returns that
    threaded value, and a contribution returning ``None`` passes it through.
    ``BROADCAST`` is what a declaration with no results sink gets — every
    contribution runs and the call returns the raw results. Only ``ADDITIVE``
    invokes the declaration body; for the other two it is documentation.
    """

    ADDITIVE = "additive"
    PIPELINE = "pipeline"
    BROADCAST = "broadcast"


class _Negated:
    """Marks a term to fold in by subtraction; see :func:`negated`."""

    __slots__ = ("term",)

    def __init__(self, term: Any) -> None:
        self.term = term


def negated(term: Any) -> Any:
    """Mark *term* to join a :func:`fold` with ``-`` instead of ``+``.

    Mirrors native's ``== source``. The bound matrices have no unary minus, so
    ``sum - term`` is the only available spelling.
    """
    return _Negated(term)


def fold(seed: Any, results: Iterable[Any]) -> Any:
    """Combine contribution *results* onto *seed*, in order: ``+`` by default,
    ``-`` for :func:`negated` results, skipping ``None`` (no opinion).

    Seeding plus the ``None`` skip is what keeps ``+ ext.terms(U)`` well-formed
    for any number of active contributions, including none.
    """
    out = seed
    for result in results:
        if result is None:
            continue
        if isinstance(result, _Negated):
            out = out - result.term
        else:
            out = out + result
    return out


class Hook:
    """One hook of an :class:`Extension` and its registered contributions.

    The handle ``@<extension>.defines`` returns: the
    ``@<model>.contributes(<hook>)`` target, and — used as a parameter
    annotation — the consumer's injection marker.
    """

    def __init__(
        self,
        extension: Extension,
        declaration: Callable[..., Any],
        kind: Kind = Kind.ADDITIVE,
    ) -> None:
        self.extension = extension
        self.name = declaration.__name__
        self.declaration = declaration
        self.kind = kind
        # One insertion-ordered relation: contribution function -> owning model.
        self._contributions: dict[Callable[..., Any], ModelSpec] = {}

    @property
    def contributions(self) -> tuple[Callable[..., Any], ...]:
        """The registered contribution functions, in registration order."""
        return tuple(self._contributions)

    def owner_of(self, contribution: Callable[..., Any]) -> ModelSpec:
        """Return the contributing model that registered *contribution*."""
        if contribution not in self._contributions:
            raise KeyError(
                f"hook '{self.extension.name}.{self.name}': {contribution!r} "
                "is not a registered contribution."
            )
        return self._contributions[contribution]

    def _register_contribution(
        self, func: Callable[..., Any], owner: ModelSpec
    ) -> Callable[..., Any]:
        """Record *func* as a contribution owned by the *owner* model."""
        self._contributions[func] = owner
        return func

    def resolve(self, ctx: Any) -> Callable[..., Any]:
        """Bind this hook to *ctx* for one injection.

        The returned callable dispatches exactly like ``ext.<hook>(...)`` on a
        :class:`BoundExtension`. This is how a hook used as a parameter
        annotation (``@<model>.interface``) reaches its consumer.
        """
        runtime_by_spec = _runtime_by_spec(ctx)

        def call(*args: Any, **kwargs: Any) -> Any:
            return call_hook(self, runtime_by_spec, ctx, args, kwargs)

        return call


class Extension:
    """A named bundle of hooks an operation module defines.

    A ``@defines`` declaration's leading parameters are the call-time arguments
    the operation passes; a **trailing parameter the call does not supply**
    receives the active contributions' results, and the body owns the combine
    rule (``min``, ``any``, :func:`fold`, …). A declaration whose every parameter
    is a call argument is a **broadcast hook**: the call returns the raw
    per-contribution results and the body is never invoked::

        momExt = Extension("momentum")

        @momExt.defines
        def terms(U: volVectorField, contributions: list[Any]) -> Any:
            return fold(zero_source(U), contributions)

        @momExt.defines
        def constrain(UEqn: fvVectorMatrix) -> None: ...  # broadcast hook

    An operation consumes the whole bundle as one injected handle by annotating
    a parameter ``Annotated[BoundExtension, <extension>]``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._hooks: dict[str, Hook] = {}

    @overload
    def defines(self, declaration: Callable[..., Any]) -> Hook: ...

    @overload
    def defines(self, *, kind: Kind) -> Callable[[Callable[..., Any]], Hook]: ...

    def defines(
        self, declaration: Callable[..., Any] | None = None, *, kind: Kind = Kind.ADDITIVE
    ) -> Hook | Callable[[Callable[..., Any]], Hook]:
        """Declare one hook of this extension; returns its :class:`Hook` handle.

        Usable bare (``@<extension>.defines``) for the default additive kind, or
        with the kind named (``@<extension>.defines(kind=Kind.PIPELINE)``).
        """
        if declaration is None:
            return lambda deferred: self._define(deferred, kind)
        return self._define(declaration, kind)

    def _define(self, declaration: Callable[..., Any], kind: Kind) -> Hook:
        hook = Hook(self, declaration, kind)
        if kind is Kind.PIPELINE and not inspect.signature(declaration).parameters:
            raise RuntimeError(
                f"Extension '{self.name}': pipeline hook '{hook.name}' declares no "
                "parameter to thread the value through."
            )
        if hook.name in self._hooks:
            raise RuntimeError(f"Extension '{self.name}': hook '{hook.name}' is already defined.")
        self._hooks[hook.name] = hook
        return hook

    @property
    def hooks(self) -> dict[str, Hook]:
        """The declared hooks by name, in declaration order."""
        return dict(self._hooks)

    def __getattr__(self, name: str) -> Hook:
        # Attribute lookup, so two extensions can share a hook name.
        if name.startswith("_"):
            raise AttributeError(name)
        hook = self._hooks.get(name)
        if hook is None:
            raise AttributeError(f"extension '{self.name}' defines no hook '{name}'")
        return hook

    def resolve(self, ctx: Any) -> BoundExtension:
        """Bind this extension to *ctx* for one injection (see BoundExtension)."""
        return BoundExtension(self, ctx)


class BoundExtension:
    """One :class:`Extension` resolved against one Context: ``ext.<hook>(...)``.

    Built fresh by :meth:`Extension.resolve` on every injection, so it never
    outlives the Context it was resolved against. A hook call runs every
    contribution whose model is active for the Context (by ``ModelSpec``
    identity), in registration order, then returns the declaration body's
    combined value — or, for a broadcast hook, the raw results list.
    """

    def __init__(self, extension: Extension, ctx: Any) -> None:
        self._extension = extension
        self._ctx = ctx
        self._runtime_by_spec = _runtime_by_spec(ctx)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        # Underscore names must miss so internal state still resolves during __init__.
        if name.startswith("_"):
            raise AttributeError(name)
        hook = self._extension._hooks.get(name)
        if hook is None:
            raise AttributeError(f"extension '{self._extension.name}' defines no hook '{name}'")

        def call(*args: Any, **kwargs: Any) -> Any:
            return call_hook(hook, self._runtime_by_spec, self._ctx, args, kwargs)

        return call


def _runtime_by_spec(ctx: Any) -> dict[Any, Any]:
    """The live ``ModelRuntime``s of *ctx*, keyed by their spec (identity)."""
    # Lazy import breaks the cycle model.extension -> model.runtime -> ... .
    from .runtime import ModelRuntime  # noqa: PLC0415

    if ctx is None:
        return {}
    return {rt.spec: rt for rt in ctx.models.values() if isinstance(rt, ModelRuntime)}


def call_hook(
    hook: Hook,
    runtime_by_spec: dict[Any, Any],
    ctx: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Dispatch one hook call: run the active contributions, then combine.

    Shared by :class:`BoundExtension` and by a hook bound on its own
    (:meth:`Hook.resolve`).
    """
    # Lazy import breaks the cycle model.extension -> model.interface -> ... .
    from .interface import _resolve_contribution_kwargs  # noqa: PLC0415

    signature = inspect.signature(hook.declaration)
    bound = signature.bind_partial(*args, **kwargs)
    call_kwargs = dict(bound.arguments)
    parameters = list(signature.parameters)
    # The results sink is the *last* parameter by convention; any other
    # unsupplied parameter is a caller mistake, not a sink. A pipeline's first
    # parameter carries the threaded value, so it can never be the sink — without
    # this a one-parameter pipeline would exempt its own argument from the guard
    # below and fail with a bare KeyError instead.
    sinkable = parameters[1:] if hook.kind is Kind.PIPELINE else parameters
    results_param = sinkable[-1] if sinkable and sinkable[-1] not in bound.arguments else None
    missing = [name for name in parameters if name not in bound.arguments and name != results_param]
    if missing:
        raise TypeError(
            f"hook '{hook.extension.name}.{hook.name}' called without "
            f"argument(s): {', '.join(missing)}"
        )
    results: list[Any] = []
    for func, owner_spec in hook._contributions.items():
        runtime = runtime_by_spec.get(owner_spec)
        if runtime is None:
            continue  # contributing model not active for this case
        resolved = _resolve_contribution_kwargs(
            f"{hook.extension.name}.{hook.name}", func, runtime, ctx, call_kwargs
        )
        value = func(**resolved)
        results.append(value)
        if hook.kind is Kind.PIPELINE and value is not None:
            # The next contribution transforms this one's output.
            call_kwargs[parameters[0]] = value
    if hook.kind is Kind.PIPELINE:
        # The threaded value *is* the answer, so the declaration body is
        # documentation only (as for broadcast). Reading it off call_kwargs
        # rather than results[-1] is what makes a trailing ``None`` — a
        # contribution with no opinion — pass the value through instead of
        # erasing it.
        return call_kwargs[parameters[0]]
    if results_param is None:
        return results  # broadcast: every parameter is a call argument
    return hook.declaration(*args, **kwargs, **{results_param: results})
