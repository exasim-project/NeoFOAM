"""
Gather contributions from many models with a model-owned interface
==================================================================

Many physics models each want to add a term to the *same* place — a source
term in an equation, a limit on the time step, a flag that stops the run. A
**model-owned interface** is that shared gather point: one model *owns* it via
``@<owner>.interface`` (the fold that decides how the pieces combine), other
models plug in with ``@<model>.contributes(<interface>)``, and a consumer
**injects the interface and calls it** to get the single combined result. New
models plug in without the owner, the consumer, or the other models changing.

This page uses the easiest possible example — **a total source term that is
the sum of per-model contributions** — and shows the things you do with an
interface: declare one on a model, gather a case's active contributors, and use
it inside an operation. ``Model(name)`` is the same factory you already use for
physics models.
"""

# %%
# 1. Declare the interface on its owning model: a name plus a fold
# ---------------------------------------------------------------
# ``@<owner>.interface`` decorates the *one* fold function that combines all
# contributions — here a plain ``sum``, so the interface means "total source
# term". The decorated name (``source``) is the interface name, and the body
# defines the empty case: no contributions => ``0.0`` (no source). The decorator
# returns a ``ModelInterface`` handle that doubles as a ``@<model>.contributes``
# target and as an operation-parameter annotation.

from typing import Iterable

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import DependencyResolver
from neofoam.framework.model import Model, ModelRuntime, bind_owned_interfaces

equation = Model("equation")


@equation.interface
def source(parts: Iterable[float]) -> float:
    return sum(parts, 0.0)  # no contribution -> 0.0


print(source.name, "empty fold =", source.fold([]))


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
# 3. Bind a case's active contributors onto the owner runtime
# -----------------------------------------------------------
# At runtime each active model has a ``ModelRuntime``. ``bind_owned_interfaces``
# is the per-case auto-wiring seam: it takes the owner runtime plus the case's
# candidate runtimes, binds the active contributors for every interface the
# owner declares, and stores each :class:`BoundModelInterface` on
# ``owner_runtime.bound_interfaces``. A bound interface is callable: calling it
# folds exactly the contributions whose model is active for this case. The
# parameter *name* (``rho``/``U``/``T``) is the lookup key into ``ctx.fields``.

ctx = Context(fields={"U": 2.0, "rho": 1.0, "T": 320.0}, models={})
equation_rt = ModelRuntime(spec=equation, name="equation", config=None)
gravity_rt = ModelRuntime(spec=gravity, name="gravity", config=None)
drag_rt = ModelRuntime(spec=drag, name="drag", config=None)
buoyancy_rt = ModelRuntime(spec=buoyancy, name="buoyancy", config=None)

bind_owned_interfaces(equation_rt, [gravity_rt, drag_rt, buoyancy_rt], ctx)
bound_source = equation_rt.bound_interfaces["source"]
# gravity = -9.81, drag = -1.0, buoyancy = 0.1*(320-300) = 2.0 -> summed -8.81
print("total source =", bound_source())


# %%
# 4. Use it in an operation: inject the interface and *call* it
# -------------------------------------------------------------
# A consumer never reads ``ctx`` for the interface — it **types a parameter with
# the interface handle itself**, and the framework injects the owner runtime's
# bound interface. The resolver finds it via ``ctx.models[<owner>.name]``, so the
# owner runtime must be registered under its model name. In a real solver the
# function below is decorated ``@<model>.operation`` and the framework resolves +
# calls it every step; here we do that by hand with the
# :class:`~neofoam.framework.dependency_resolver.DependencyResolver`. This module
# must **not** use ``from __future__ import annotations`` (it would stringify the
# annotation and hide the handle from the resolver).

ctx.models["equation"] = equation_rt  # the resolver looks up ctx.models["equation"]
resolver = DependencyResolver()

DT = 0.01


# In a solver: @momentum.operation
def advance_velocity(U: float, src: source) -> float:  # type: ignore[valid-type]
    # src() gathers every active model's contribution into one total source term
    return U + DT * src()


resolved = resolver.resolve_arguments(advance_velocity, ctx)
print("injected", type(resolved["src"]).__name__, "-> src() =", resolved["src"]())
print("advance_velocity(**resolved) =", advance_velocity(**resolved))


# %%
# 5. Add a model with zero changes elsewhere (the whole point)
# ------------------------------------------------------------
# Adding a new physics model is *only* a new ``@contributes`` plus including its
# runtime in the case — the fold, the operation, and the existing models are
# untouched. Register a porous resistance and re-bind the owner runtime with the
# new candidate; the same ``advance_velocity`` now sees the extra term, because
# it asked the interface for "the total", not for specific models.

porous = Model("porous")


@porous.contributes(source)
def porous_force(U: float) -> float:
    return -2.0 * U


porous_rt = ModelRuntime(spec=porous, name="porous", config=None)
bind_owned_interfaces(equation_rt, [gravity_rt, drag_rt, buoyancy_rt, porous_rt], ctx)
print("total source after adding 'porous' =", equation_rt.bound_interfaces["source"]())


# %%
# 6. Turn a model off: leave its runtime out of the case
# ------------------------------------------------------
# A model contributes iff its runtime is among the case's active contributors.
# There is no global active-set to toggle: re-bind with a candidate list that
# omits ``porous`` and its term simply drops out of the fold — exactly how an
# optional model whose config is absent stops contributing.

bind_owned_interfaces(equation_rt, [gravity_rt, drag_rt, buoyancy_rt], ctx)
print("with 'porous' off =", equation_rt.bound_interfaces["source"]())


# %%
# What if a model needs something the case doesn't have?
# ------------------------------------------------------
# The interface fails **loudly** rather than silently dropping a term: a
# contribution whose parameter nothing supplies raises a ``ValueError`` when the
# interface is folded, naming the interface, the contribution, and the missing
# parameter. A misconfigured model can never silently vanish from the sum.

needs_phi = Model("needsPhi")


@needs_phi.contributes(source)
def needs_phi_force(phi: float) -> float:  # 'phi' is not in ctx.fields
    return phi


needs_phi_rt = ModelRuntime(spec=needs_phi, name="needsPhi", config=None)
bind_owned_interfaces(
    equation_rt, [gravity_rt, drag_rt, buoyancy_rt, needs_phi_rt], ctx
)
try:
    equation_rt.bound_interfaces["source"]()
except ValueError as err:
    print("fold raised:", err)


# %%
# See also
# --------
#
# - :doc:`/explanation/parameter-injection` — how a contribution's parameters
#   are resolved from the Context.
# - :doc:`/auto_how-to/example_use_depends_for_injection` — every injection
#   pattern a contribution can use (fields, models, ``Depends``).
# - :doc:`/auto_how-to/example_register_a_model` — the ``Model(name)`` pattern,
#   and how models are registered.
