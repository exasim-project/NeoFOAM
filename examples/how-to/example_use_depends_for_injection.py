"""
Inject dependencies into operations
===================================

Operations and the solver initializer don't receive a raw
:class:`Context` — they declare what they need as typed parameters and
the framework fills them in. This page walks through the five
injection patterns the resolver understands, executing each one so
the resolved values show up below the code.
"""

# %%
# Set the stage: build a Context the resolver can read from
# ---------------------------------------------------------
# The :class:`DependencyResolver` is the runtime piece that turns a
# function signature into a kwargs dict.  It reads ``ctx.fields``,
# ``ctx.models``, and any ``Depends`` markers attached to parameter
# annotations. Build a populated :class:`Context` and a resolver
# once — every example below reuses them.

import logging
from typing import Annotated, Any

from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import DependencyResolver
from neofoam.framework.initialization import Depends


class _Algorithm:
    """Stand-in for a solver-side object stored under ``ctx.models``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def solve(self) -> str:
        return f"{self.name}.solve() ran"


ctx = Context(
    fields={"U": 0.5, "p": 0.2, "T": 300.0},
    models={"algorithm": _Algorithm("PISO")},
)
resolver = DependencyResolver()

print("ctx.fields =", dict(ctx.fields))
print("ctx.models =", {k: type(v).__name__ for k, v in ctx.models.items()})


# %%
# 1) Inject a context field by parameter name
# -------------------------------------------
# Plain-typed parameters whose **names appear in ``ctx.fields``** are
# pulled from there automatically. The parameter name *is* the lookup
# key — rename the parameter and the lookup changes with it.


def advance(U: float, p: float) -> float:
    return U + 1e-3 * p


resolved = resolver.resolve_arguments(advance, ctx)
print("resolved kwargs:", resolved)
print("advance(**resolved) =", advance(**resolved))


# %%
# 2) Inject from a specific container with Annotated[T, "models"]
# ---------------------------------------------------------------
# Use ``Annotated[T, "<container>"]`` when you want to pull from a
# container *other than* ``fields``. The string marker names the
# container attribute on :class:`Context` (``models``, ``fields``,
# or any other dict-typed attribute); the parameter name is still
# the key within it.


def correct(algorithm: Annotated[Any, "models"]) -> str:
    return algorithm.solve()


resolved = resolver.resolve_arguments(correct, ctx)
print("resolved kwargs:", resolved)
print("correct(**resolved) =", correct(**resolved))


# %%
# 3) Inject from an arbitrary provider via Depends(callable)
# ----------------------------------------------------------
# When the value doesn't live in ``Context``, mark the parameter with
# :class:`Depends` and hand it a provider callable. The resolver calls
# the provider on first request and caches the result (default scope:
# ``time_step``). Pass ``Depends(provider, cache=False)`` to re-run
# every call, or ``Depends(provider, scope="iteration")`` to bind the
# cache lifetime to the inner iteration loop instead.


def make_logger() -> logging.Logger:
    logger = logging.getLogger("neofoam.demo")
    logger.setLevel(logging.INFO)
    return logger


def announce(logger: Annotated[Any, Depends(make_logger)]) -> str:
    return f"got logger named {logger.name!r}"


resolved = resolver.resolve_arguments(announce, ctx)
print("announce(**resolved) =", announce(**resolved))


# %%
# 4) Reach into the Context by path
# ---------------------------------
# :class:`Depends` also accepts a **string path** that's resolved
# against the :class:`Context`. The path is dotted: ``"fields.U"``,
# ``"models.algorithm"``, or any attribute on ``Context``. Useful
# when the parameter name and the field key disagree (e.g. you want
# a parameter called ``u_field`` to pull ``ctx.fields["U"]``).


def report(u_field: Annotated[Any, Depends("fields.U")]) -> str:
    return f"u_field = {u_field}"


resolved = resolver.resolve_arguments(report, ctx)
print("report(**resolved) =", report(**resolved))


# %%
# 5) Caching across calls within a scope
# --------------------------------------
# A ``Depends(callable)`` is invoked at most once per scope per
# resolver. Watch the counter: even though ``count_calls`` is
# requested three times, the provider only runs once because the
# default ``cache=True`` and the cache for ``"time_step"`` is shared.

_call_log: list[int] = []


def count_calls() -> int:
    _call_log.append(1)
    return len(_call_log)


def needs_counter(counter: Annotated[int, Depends(count_calls)]) -> int:
    return counter


for i in range(3):
    kwargs = resolver.resolve_arguments(needs_counter, ctx)
    print(f"call {i}: counter = {kwargs['counter']}, provider ran {sum(_call_log)}x")

# Drop the time_step cache and the provider runs again on the next call.
resolver.clear_scope("time_step")
kwargs = resolver.resolve_arguments(needs_counter, ctx)
print(
    f"after clear_scope: counter = {kwargs['counter']}, provider ran {sum(_call_log)}x"
)


# %%
# See also
# --------
#
# - :doc:`/explanation/parameter-injection` — the resolution algorithm
#   in detail.
# - :doc:`/reference/framework/dependency_resolver` — full API.
