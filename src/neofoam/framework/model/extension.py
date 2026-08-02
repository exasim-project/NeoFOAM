# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Extension — a bundle of hooks an operation module declares for models to extend.

An operation module declares one :class:`Extension` per extensible operation and
one ``@<extension>.defines`` function per :class:`Hook`: the point the operation
calls, the arguments it passes, and — via a trailing parameter the call does not
supply — what to do with the contributions' results. Models register plain
functions on a hook with ``@<model>.contributes(<hook>)``; an operation receives
the whole bundle as one injected :class:`BoundExtension` handle and calls each
hook once, exactly where native calls it.

A contribution is active iff its model is active for the case (a live
``ModelRuntime`` in ``ctx.models``), so the operations never name a model and a
case activates contributions by its files alone.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from .spec import ModelSpec


class _Negated:
    """Marks a term to fold in by subtraction; see :func:`negated`."""

    __slots__ = ("term",)

    def __init__(self, term: Any) -> None:
        self.term = term


def negated(term: Any) -> Any:
    """Mark *term* to join a :func:`fold` with ``-`` instead of ``+``.

    Native's ``== source`` moves a source to the right-hand side, i.e. subtracts
    it from the assembled matrix; a contribution returns ``negated(source)``
    to fold it in with exactly that arithmetic (``sum - source``, not
    ``sum + (-source)`` — the bound matrices have no unary minus).
    """
    return _Negated(term)


def fold(seed: Any, results: Iterable[Any]) -> Any:
    """Combine contribution *results* onto *seed*, in order: ``+`` by default,
    ``-`` for :func:`negated` results, skipping ``None`` (no opinion).

    The combine rule for equation-term hooks — a hook declaration's body calls
    it on the results it receives (``return fold(zero_source(U), contributions)``)
    so ``+ ext.terms(U)`` is well-formed with any number of active
    contributions, including none.
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
    """One hook of an :class:`Extension`, as declared by ``@<extension>.defines``.

    Holds the declaration function (its ``__name__`` is the hook name, its
    leading parameters are the call-time arguments, a trailing parameter takes
    the contribution results — see :class:`Extension`) and the registered
    contributions. The handle is the ``@<model>.contributes(<hook>)`` target.
    """

    def __init__(self, extension: Extension, declaration: Callable[..., Any]) -> None:
        self.extension = extension
        self.name = declaration.__name__
        self.declaration = declaration
        # One insertion-ordered relation: contribution function -> owning model.
        self._contributions: dict[Callable[..., Any], ModelSpec] = {}

    @property
    def contributions(self) -> tuple[Callable[..., Any], ...]:
        """The registered contribution functions, in registration order."""
        return tuple(self._contributions)

    def owner_of(self, contribution: Callable[..., Any]) -> ModelSpec:
        """Return the contributing model that registered *contribution*.

        Raises:
            KeyError: if *contribution* was never registered on this hook.
        """
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
        :class:`BoundExtension` — it runs the active contributions and returns
        the declaration body's combined value. This is how a model-owned
        interface (``@<model>.interface``) is injected: the resolver binds the
        annotating hook to the live Context, and the consumer just calls it.
        """
        runtime_by_spec = _runtime_by_spec(ctx)

        def call(*args: Any, **kwargs: Any) -> Any:
            return call_hook(self, runtime_by_spec, ctx, args, kwargs)

        return call


class Extension:
    """A named bundle of hooks an operation module defines.

    Declared once at module import next to the operations it serves; each
    ``@defines`` function declares one :class:`Hook`. Its leading parameters are
    the call-time arguments the operation passes; a **trailing parameter the
    call does not supply** receives the list of active contributions' results,
    and the body combines them — it owns the rule (``min``, ``any``,
    :func:`fold`, …)::

        momExt = Extension("momentum")

        @momExt.defines
        def terms(U: volVectorField, contributions: list[Any]) -> Any:
            return fold(zero_source(U), contributions)

        @momExt.defines
        def constrain(UEqn: fvVectorMatrix) -> None: ...  # broadcast hook

    A declaration whose every parameter is a call argument is a **broadcast
    hook**: the call returns the raw per-contribution results (the body is
    never invoked) so the operation can inspect or ignore them.

    Models contribute per hook with ``@<model>.contributes(<hook>)``: a
    contribution's parameters matching the hook's call-time arguments are taken
    from the call, and the rest resolve from the contributing model's own config
    plus the Context. An operation consumes the whole bundle as one injected
    handle by annotating a parameter ``Annotated[BoundExtension, <extension>]``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._hooks: dict[str, Hook] = {}

    def defines(self, declaration: Callable[..., Any]) -> Hook:
        """Declare one hook of this extension (see the class docstring).

        Returns the :class:`Hook` handle under the declaration's name, the
        target models pass to ``@<model>.contributes(<hook>)``.
        """
        hook = Hook(self, declaration)
        if hook.name in self._hooks:
            raise RuntimeError(f"Extension '{self.name}': hook '{hook.name}' is already defined.")
        self._hooks[hook.name] = hook
        return hook

    @property
    def hooks(self) -> dict[str, Hook]:
        """The declared hooks by name, in declaration order."""
        return dict(self._hooks)

    def __getattr__(self, name: str) -> Hook:
        # Hook handles are reachable as attributes (``momExt.terms``) so two
        # extensions can share a hook name without colliding at module level.
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
    contribution whose model is active for the Context, in registration order
    (activation by ``ModelSpec`` identity, as everywhere else), then:

    * the declaration has a trailing parameter the call did not supply — the
      results list is passed there, and the **body's combined value** is the
      hook's result (no active contribution -> the body sees ``[]``);
    * every declaration parameter is a call argument (a broadcast hook) — the
      raw per-contribution results are returned as a list, so the operation
      can inspect them or ignore them (``ext.correct(U)``).
    """

    def __init__(self, extension: Extension, ctx: Any) -> None:
        self._extension = extension
        self._ctx = ctx
        self._runtime_by_spec = _runtime_by_spec(ctx)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        # Underscore names never dispatch: internal state must miss naturally
        # (also keeps this re-entrant while __init__ has not run yet).
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

    The shared engine behind :class:`BoundExtension` (and the model-owned
    interface sugar): binds the call arguments against the hook declaration,
    resolves and runs each active contribution in registration order, and either
    hands the results to the declaration's trailing parameter (its body
    combines) or returns them raw (a broadcast hook — every parameter bound).
    """
    # Lazy import breaks the cycle model.extension -> model.interface -> ... .
    from .interface import _resolve_contribution_kwargs  # noqa: PLC0415

    signature = inspect.signature(hook.declaration)
    bound = signature.bind_partial(*args, **kwargs)
    call_kwargs = dict(bound.arguments)
    parameters = list(signature.parameters)
    # By convention the results sink is the *last* declaration parameter; any
    # other unsupplied parameter is a caller mistake, not a sink.
    results_param = parameters[-1] if parameters and parameters[-1] not in bound.arguments else None
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
        results.append(func(**resolved))
    if results_param is None:
        return results  # broadcast: every parameter is a call argument
    return hook.declaration(*args, **kwargs, **{results_param: results})
