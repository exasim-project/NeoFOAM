"""
Gather contributions from many models with an Interface
=======================================================

Many physics models each want to add a term to the *same* place — a source
term in an equation, a limit on the time step, a flag that stops the run.
An ``Interface`` is that shared **gather point**: each model registers a
small ``@contribute`` function, one ``@combine`` fold decides how the pieces
combine, and a consumer **injects the interface and calls it** to get the
single combined result. New models plug in without the consumer or the other
models changing.

This page uses the easiest possible example — **a total source term that is
the sum of per-model contributions** — and shows the three things you do with
an interface: build one, gather several models into it, and use it inside an
operation. ``Interface(name)`` mirrors ``Model(name)``.
"""

# %%
# 1. Build the interface: a name plus a fold
# ------------------------------------------
# ``Interface(name)`` is the gather point. ``@<iface>.combine`` registers the
# *one* function that combines all contributions — here a plain ``sum``, so
# the interface means "total source term". The fold also defines the empty
# case: no contributions ⇒ ``0.0`` (no source). Only one ``@combine`` is
# allowed per interface.

from typing import Iterable

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import DependencyResolver
from neofoam.framework.interface import Interface

source = Interface("source")


@source.combine
def total(parts: Iterable[float]) -> float:
    return sum(parts, 0.0)  # no contribution -> 0.0


print(source.name, "fold of [] =", total([]))


# %%
# 2. Gather several models: each adds one ``@contribute``
# -------------------------------------------------------
# Every model registers an **operation-style** contribution: it declares the
# fields it needs as typed parameters (resolved from the
# :class:`~neofoam.framework.context.Context` exactly like ``@model.operation``)
# and returns its one contribution to the sum. The three functions below stand
# in for three independent models — **none knows the others exist**, and none
# touches ``ctx`` directly. A contribution may *not* declare a ``Context``
# parameter.


@source.contribute
def gravity(rho: float) -> float:
    return rho * -9.81  # body force


@source.contribute
def drag(U: float) -> float:
    return -0.5 * U  # linear resistance


@source.contribute
def buoyancy(T: float) -> float:
    return 0.1 * (T - 300.0)  # thermal source


print("models contributing to 'source':", [c.__name__ for c in source._contributions])


# %%
# 3. Combine them: ``collect(ctx)`` folds over a Context
# ------------------------------------------------------
# ``collect(ctx)`` resolves each contribution's parameters from the Context,
# calls it, and runs them through the fold. The parameter *name* is the lookup
# key into ``ctx.fields``. With the values below: gravity ``= -9.81``, drag
# ``= -1.0``, buoyancy ``= 0.1*(320-300) = 2.0`` — summed to ``-8.81``.

ctx = Context(fields={"U": 2.0, "rho": 1.0, "T": 320.0}, models={})
print("total source =", source.collect(ctx))


# %%
# 4. Use it in an operation: inject the interface and *call* it
# -------------------------------------------------------------
# A consumer never reads ``ctx`` for the interface — it **types a parameter
# with the interface itself**, and the framework injects a ``BoundInterface``
# whose call runs ``collect(ctx)``. In a real solver the function below is
# decorated ``@<model>.operation`` and the framework resolves + calls it every
# step; here we do that by hand with the
# :class:`~neofoam.framework.dependency_resolver.DependencyResolver` to show
# the same thing. The interface must sit in ``ctx.interfaces`` under its name,
# and this module must **not** use ``from __future__ import annotations`` (it
# would stringify the annotation and hide the spec from the resolver).

ctx = Context(
    fields={"U": 2.0, "rho": 1.0, "T": 320.0},
    models={},
    interfaces={"source": source},  # keyed by the interface's name
)
resolver = DependencyResolver()

DT = 0.01


# In a solver: @momentum.operation
def advance_velocity(U: float, src: source) -> float:
    # src() gathers every model's contribution into one total source term
    return U + DT * src()


resolved = resolver.resolve_arguments(advance_velocity, ctx)
print("injected", type(resolved["src"]).__name__, "-> src() =", resolved["src"]())
print("advance_velocity(**resolved) =", advance_velocity(**resolved))


# %%
# 5. Add a model with zero changes elsewhere (the whole point)
# ------------------------------------------------------------
# Adding a new physics model is *only* a new ``@contribute`` — the fold, the
# operation, and the existing models are untouched. Register a porous
# resistance and the same ``advance_velocity`` now sees the new term, because
# it asked the interface for "the total", not for specific models.


@source.contribute
def porous(U: float) -> float:
    return -2.0 * U


print("total source after adding 'porous' =", source.collect(ctx))
resolved = resolver.resolve_arguments(advance_velocity, ctx)
print("advance_velocity now =", advance_velocity(**resolved))


# %%
# 6. Turn a model off
# -------------------
# Contributions are active by default. ``deactivate`` drops one from the fold
# without unregistering it (e.g. a model whose config is absent); ``activate``
# restores it. This is how an optional model that is switched off simply stops
# contributing.

source.deactivate(porous)
print("with 'porous' off  =", source.collect(ctx))
source.activate(porous)
print("with 'porous' on   =", source.collect(ctx))


# %%
# What if a model needs something the case doesn't have?
# ------------------------------------------------------
# The interface fails **loudly** rather than silently dropping a term: a
# contribution whose parameter nothing supplies raises a ``ValueError`` when
# the interface is folded, naming the interface, the contribution, and the
# missing parameter. A misconfigured model can never silently vanish from the
# sum.


@source.contribute
def needs_phi(phi: float) -> float:  # 'phi' is not in ctx.fields
    return phi


try:
    source.collect(ctx)
except ValueError as err:
    print("collect raised:", err)
finally:
    source.deactivate(needs_phi)  # keep re-runs of this page green


# %%
# See also
# --------
#
# - :doc:`/explanation/parameter-injection` — how a contribution's parameters
#   are resolved from the Context.
# - :doc:`/auto_how-to/example_use_depends_for_injection` — every injection
#   pattern a contribution can use (fields, models, ``Depends``).
# - :doc:`/auto_how-to/example_register_a_model` — the ``Model(name)`` pattern
#   that ``Interface(name)`` mirrors, and how models are registered.
