# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""ExtensionPoint — a multi-method extension seam declared by an operation module.

An operation module declares one ``ExtensionPoint`` naming the interface class it
accepts; that class defines one method per site the operations can be extended at,
each with a no-op default. Models register implementation factories on the point via
``@<model>.extends(<point>)``, and an operation receives every active implementation
as a single injected ``Extensions`` container by annotating a parameter with the
point.

The container acts as one aggregated implementation: ``ext.constrain(UEqn)`` calls
the site on every active implementation in registration order, and a term site
folds directly into an equation expression — ``... + ext.terms(U)`` adds every
active model's terms (a ``negated`` term joins with ``-``) and leaves the sum
untouched when no model is active. Sites whose per-implementation return values
the operation must inspect (a transformed value, a handled flag) are iterated
explicitly instead.

Use this where the sites are several and heterogeneous (a term to add here, a
constraint to apply there). Where the extension is a single value combined by one
rule, use ``ModelInterface`` (``.interface`` / ``.contributes``) instead — it folds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, Iterable, Iterator, TypeVar

from .interface import _resolve_contribution_kwargs

if TYPE_CHECKING:
    from .spec import ModelSpec

T = TypeVar("T")


class _Negated:
    """Marks a term to fold in by subtraction; see :func:`negated`."""

    __slots__ = ("term",)

    def __init__(self, term: Any) -> None:
        self.term = term


def negated(term: Any) -> Any:
    """Mark *term* to join a ``+ ext.<site>(...)`` fold with ``-`` instead of ``+``.

    Native's ``== source`` moves a source to the right-hand side, i.e. subtracts
    it from the assembled matrix; an implementation returns ``negated(source)``
    to fold it in with exactly that arithmetic (``sum - source``, not
    ``sum + (-source)`` — the bound matrices have no unary minus).
    """
    return _Negated(term)


def _make_term_fold(sum_type: type) -> type:
    """The carrier class for one point's term folds, subclassing *sum_type*.

    The pybFoam operators raise ``TypeError`` for a foreign right operand instead
    of returning ``NotImplemented``, so a plain ``__radd__`` object on the right of
    ``+`` is never consulted. Subclassing the sum's own type is the one hook left:
    Python tries the *right* operand's reflected method first when its type is a
    proper subclass of the left's. The carrier therefore never calls
    ``sum_type.__init__`` and holds no C++ payload — it must only ever stand on
    the right of ``+`` and die inside the expression.
    """

    class _TermFold(sum_type):  # type: ignore[misc]
        def __init__(self, terms: list[Any]) -> None:
            self._terms = terms

        def __radd__(self, lhs: Any) -> Any:
            out = lhs
            for term in self._terms:
                if isinstance(term, _Negated):
                    out = out - term.term
                else:
                    out = out + term
            return out

    return _TermFold


class Extensions(Generic[T]):
    """The active extension implementations for one injection, in registration order.

    Built fresh by :meth:`ExtensionPoint.resolve` on every injection, so it never
    outlives the Context it was resolved against and never keeps mesh-bound objects
    alive across runs.

    Beyond iteration, ``len`` and truthiness, the container dispatches any site of
    the point's interface across all implementations at once: ``ext.constrain(UEqn)``
    calls ``constrain`` on each implementation in registration order, and on a point
    declared with ``folds_into`` a term site joins an equation expression directly —
    every returned term folds into the running sum in registration order, ``+`` by
    default, ``-`` for :func:`negated` terms, and an empty site leaves the sum
    untouched (it returns the identical object, so inactivity costs nothing).
    Sites whose per-implementation return value the operation must inspect are
    iterated explicitly instead.

    Example::

        ext.correct_boundary_velocity(U)
        UEqn = fvVectorMatrix(momentum_sum + ext.terms(U))
    """

    def __init__(
        self,
        instances: Iterable[T],
        protocol: type[T] | None = None,
        fold_type: type | None = None,
    ) -> None:
        self._instances = list(instances)
        self._protocol = protocol
        self._fold_type = fold_type

    def __iter__(self) -> Iterator[T]:
        return iter(self._instances)

    def __len__(self) -> int:
        return len(self._instances)

    def __bool__(self) -> bool:
        return bool(self._instances)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        # Underscore names never dispatch: internal state must miss naturally
        # (also keeps this re-entrant while __init__ has not run yet).
        if name.startswith("_"):
            raise AttributeError(name)
        protocol = self._protocol
        if protocol is not None and not callable(getattr(protocol, name, None)):
            raise AttributeError(f"'{protocol.__name__}' declares no extension site '{name}'")
        instances = self._instances
        fold_type = self._fold_type

        def site(*args: Any, **kwargs: Any) -> Any:
            terms: list[Any] = []
            for instance in instances:
                result = getattr(instance, name)(*args, **kwargs)
                if result is not None:
                    terms.extend(result)
            return fold_type(terms) if fold_type is not None else None

        return site


class ExtensionPoint(Generic[T]):
    """A named extension seam an operation module declares: the interface it accepts.

    Declared once at module import next to the operations it serves; models then
    register factories on it with ``@<model>.extends(<point>)``. Annotate an
    operation parameter ``Annotated[Extensions[Iface], <point>]`` and the resolver
    injects the active implementations. The point holds only factories and their
    owning specs — no Context and no live case objects — so a single module-level
    instance is safe across runs.

    ``folds_into`` names the type term sums have at the operations' fold sites
    (``+ ext.terms(U)``); declare it iff the interface has term sites. See
    :func:`_make_term_fold` for why the exact type matters.

    Example::

        momentum_extension = ExtensionPoint(
            "momentum_extension", MomentumExtension, folds_into=pyf.tmp_fvVectorMatrix
        )
    """

    def __init__(self, name: str, protocol: type[T], folds_into: type | None = None) -> None:
        self.name = name
        self.protocol = protocol
        self._fold_type = _make_term_fold(folds_into) if folds_into is not None else None
        # One insertion-ordered relation: factory function -> registering model.
        self._factories: dict[Callable[..., T], ModelSpec] = {}

    @property
    def factories(self) -> tuple[Callable[..., T], ...]:
        """The registered implementation factories, in registration order."""
        return tuple(self._factories)

    def owner_of(self, factory: Callable[..., T]) -> ModelSpec:
        """Return the model that registered *factory*.

        Raises:
            KeyError: if *factory* was never registered on this extension point.
        """
        if factory not in self._factories:
            raise KeyError(
                f"extension point '{self.name}': {factory!r} is not a registered factory."
            )
        return self._factories[factory]

    def resolve(self, ctx: Any) -> Extensions[T]:
        """Build the implementations active for *ctx*, in registration order.

        A factory is active iff its registering model has a live ``ModelRuntime``
        among ``ctx.models`` whose ``spec`` **is** that model (identity, as in
        ``active_contributors`` — the Context key is the runtime's instance name,
        which a model with an instance id does not share with its spec). Each
        active factory's parameters are resolved against its own runtime's config
        plus *ctx*, exactly as an ``@model.operation`` body's are.
        """
        # Lazy import breaks the cycle model.extension -> model.runtime -> ... .
        from .runtime import ModelRuntime  # noqa: PLC0415

        runtime_by_spec = (
            {rt.spec: rt for rt in ctx.models.values() if isinstance(rt, ModelRuntime)}
            if ctx is not None
            else {}
        )
        instances: list[T] = []
        for factory, owner_spec in self._factories.items():
            runtime = runtime_by_spec.get(owner_spec)
            if runtime is None:
                continue  # registering model not active for this case
            kwargs = _resolve_contribution_kwargs(self.name, factory, runtime, ctx)
            instances.append(factory(**kwargs))
        return Extensions(instances, protocol=self.protocol, fold_type=self._fold_type)

    def _register_factory(self, func: Callable[..., T], owner: ModelSpec) -> Callable[..., T]:
        """Record *func* as an implementation factory owned by the *owner* model."""
        self._factories[func] = owner
        return func
