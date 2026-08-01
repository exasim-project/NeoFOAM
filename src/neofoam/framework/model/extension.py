# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""ExtensionPoint — a multi-method extension seam declared by an operation module.

An operation module declares one ``ExtensionPoint`` naming the interface class it
accepts; that class defines one method per site the operations can be extended at,
each with a no-op default. Models register implementation factories on the point via
``@<model>.extends(<point>)``, and an operation receives every active implementation
as a single injected ``Extensions`` container by annotating a parameter with the
point.

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


class Extensions(Generic[T]):
    """The active extension implementations for one injection, in registration order.

    Built fresh by :meth:`ExtensionPoint.resolve` on every injection, so it never
    outlives the Context it was resolved against and never keeps mesh-bound objects
    alive across runs. It deliberately offers only iteration, ``len`` and truthiness:
    operations call the extension methods in explicit loops at the exact sites.

    Example::

        for extension in extensions:
            extension.correct(U)
    """

    def __init__(self, instances: Iterable[T]) -> None:
        self._instances = list(instances)

    def __iter__(self) -> Iterator[T]:
        return iter(self._instances)

    def __len__(self) -> int:
        return len(self._instances)

    def __bool__(self) -> bool:
        return bool(self._instances)


class ExtensionPoint(Generic[T]):
    """A named extension seam an operation module declares: the interface it accepts.

    Declared once at module import next to the operations it serves; models then
    register factories on it with ``@<model>.extends(<point>)``. Annotate an
    operation parameter ``Annotated[Extensions[Iface], <point>]`` and the resolver
    injects the active implementations. The point holds only factories and their
    owning specs — no Context and no live case objects — so a single module-level
    instance is safe across runs.

    Example::

        momentum_extension = ExtensionPoint(
            "momentum_extension", MomentumExtension
        )
    """

    def __init__(self, name: str, protocol: type[T]) -> None:
        self.name = name
        self.protocol = protocol
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
        return Extensions(instances)

    def _register_factory(self, func: Callable[..., T], owner: ModelSpec) -> Callable[..., T]:
        """Record *func* as an implementation factory owned by the *owner* model."""
        self._factories[func] = owner
        return func
