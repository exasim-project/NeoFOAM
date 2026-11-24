"""
Gather contributions from many models with a model-owned interface
==================================================================

Many physics models each want to add a term to the *same* place — a source
term in an equation, a limit on the time step, a flag that stops the run. A
**model-owned interface** is that shared gather point: one model *owns* it via
``@<owner>.interface`` (the body that decides how the pieces combine), other
models plug in with ``@<model>.contributes(<interface>)``, and a consumer
**injects the interface and calls it** to get the single combined result. New
models plug in without the owner, the consumer, or the other models changing.

An interface is an extension :class:`~neofoam.framework.model.extension.Hook`
on a model-private extension — the same mechanism behind the operation-declared
seams of :doc:`/how-to/extend-operations`, in the model-owned spelling.

This page uses the easiest possible example — **a total source term that is
the sum of per-model contributions** — and shows the things you do with an
interface: declare one on a model, activate a case's contributors, and use
it inside an operation. ``Model(name)`` is the same factory you already use for
physics models.
"""

# %%
# 1. Declare the interface on its owning model: a name plus a combine rule
# ------------------------------------------------------------------------
# ``@<owner>.interface`` decorates the *one* function that combines all
# contributions — its parameter receives the list of active contributions'
# results, and its body owns the rule: here a plain ``sum``, so the interface
# means "total source term", and the empty case is ``0.0`` (no source). The
# decorator returns a ``Hook`` handle that doubles as a ``@<model>.contributes``
# target and as an operation-parameter annotation.

from typing import Iterable

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import DependencyResolver
from neofoam.framework.model import Model, ModelRuntime

equation = Model("equation")


@equation.interface
def source(parts: Iterable[float]) -> float:
    return sum(parts, 0.0)  # no contribution -> 0.0


print(source.name, "empty fold =", source.resolve(Context(fields={}, models={}))())


# %%
# 2. Gather several models: each adds one ``@contributes``
# --------------------------------------------------------
# Every model registers an **operation-style** contribution: it declares the
# fields it needs as typed parameters (resolved from the
# :class:`~neofoam.framework.context.Context` exactly like ``@model.operation``)
# and returns its one contribution to the sum. The three models below stand in
# for three independent physics models — **none knows the others exist**, and
# none touches ``ctx`` directly.

gravity = Model("gravity")
drag = Model("drag")
buoyancy = Model("buoyancy")


@gravity.contributes(source)
def gravity_force(rho: float) -> float:
    return rho * -9.81  # body force


@drag.contributes(source)
def drag_force(U: float) -> float:
    return -0.5 * U  # linear resistance


@buoyancy.contributes(source)
def buoyancy_force(T: float) -> float:
    return 0.1 * (T - 300.0)  # thermal source


print("models contributing to 'source':", [c.__name__ for c in source.contributions])


# %%
# 3. Activation is the Context: a contribution folds iff its runtime is there
# ---------------------------------------------------------------------------
# At runtime each active model has a ``ModelRuntime`` registered in
# ``ctx.models``. ``source.resolve(ctx)`` binds the interface to that Context;
# the returned handle is callable, and calling it runs exactly the
# contributions whose model is active and hands their results to the declared
# combine. The parameter *name* (``rho``/``U``/``T``) is the lookup key into
# ``ctx.fields``.

gravity_rt = ModelRuntime(spec=gravity, name="gravity", config=None)
drag_rt = ModelRuntime(spec=drag, name="drag", config=None)
buoyancy_rt = ModelRuntime(spec=buoyancy, name="buoyancy", config=None)

ctx = Context(
    fields={"U": 2.0, "rho": 1.0, "T": 320.0},
    models={"gravity": gravity_rt, "drag": drag_rt, "buoyancy": buoyancy_rt},
)
# gravity = -9.81, drag = -1.0, buoyancy = 0.1*(320-300) = 2.0 -> summed -8.81
print("total source =", source.resolve(ctx)())


# %%
# 4. Use it in an operation: inject the interface and *call* it
# -------------------------------------------------------------
# A consumer never reads ``ctx`` for the interface — it **types a parameter with
# the interface handle itself**, and the framework injects the handle bound to
# the live Context. In a real solver the function below is decorated
# ``@<model>.operation`` and the framework resolves + calls it every step; here
# we do that by hand with the
# :class:`~neofoam.framework.dependency_resolver.DependencyResolver`. This module
# must **not** use ``from __future__ import annotations`` (it would stringify the
# annotation and hide the handle from the resolver).

resolver = DependencyResolver()

DT = 0.01


# In a solver: @momentum.operation
def advance_velocity(U: float, src: source) -> float:  # type: ignore[valid-type]
    # src() gathers every active model's contribution into one total source term
    return U + DT * src()


resolved = resolver.resolve_arguments(advance_velocity, ctx)
print("injected src() =", resolved["src"]())
print("advance_velocity(**resolved) =", advance_velocity(**resolved))


# %%
# 5. Add a model with zero changes elsewhere (the whole point)
# ------------------------------------------------------------
# Adding a new physics model is *only* a new ``@contributes`` plus its runtime
# in the case — the combine rule, the operation, and the existing models are
# untouched. Register a porous resistance and include its runtime; the same
# ``advance_velocity`` now sees the extra term, because it asked the interface
# for "the total", not for specific models.

porous = Model("porous")


@porous.contributes(source)
def porous_force(U: float) -> float:
    return -2.0 * U


ctx.models["porous"] = ModelRuntime(spec=porous, name="porous", config=None)
print("total source after adding 'porous' =", source.resolve(ctx)())


# %%
# 6. Turn a model off: leave its runtime out of the case
# ------------------------------------------------------
# A model contributes iff its runtime is in ``ctx.models``. There is no global
# active-set to toggle: drop ``porous`` from the Context and its term simply
# drops out of the fold — exactly how an optional model whose config is absent
# stops contributing.

del ctx.models["porous"]
print("with 'porous' off =", source.resolve(ctx)())


# %%
# What if a model needs something the case doesn't have?
# ------------------------------------------------------
# The interface fails **loudly** rather than silently dropping a term: a
# contribution whose parameter nothing supplies raises a ``ValueError`` when the
# interface is called, naming the interface, the contribution, and the missing
# parameter. A misconfigured model can never silently vanish from the sum.

needs_phi = Model("needsPhi")


@needs_phi.contributes(source)
def needs_phi_force(phi: float) -> float:  # 'phi' is not in ctx.fields
    return phi


ctx.models["needsPhi"] = ModelRuntime(spec=needs_phi, name="needsPhi", config=None)
try:
    source.resolve(ctx)()
except ValueError as err:
    print("fold raised:", err)


# %%
# See also
# --------
#
# - :doc:`/how-to/extend-operations` — the operation-owned spelling of the same
#   mechanism: an ``Extension`` bundling several hooks.
# - :doc:`/explanation/parameter-injection` — how a contribution's parameters
#   are resolved from the Context.
# - :doc:`/auto_how-to/example_use_depends_for_injection` — every injection
#   pattern a contribution can use (fields, models, ``Depends``).
# - :doc:`/auto_how-to/example_register_a_model` — the ``Model(name)`` pattern,
#   and how models are registered.
