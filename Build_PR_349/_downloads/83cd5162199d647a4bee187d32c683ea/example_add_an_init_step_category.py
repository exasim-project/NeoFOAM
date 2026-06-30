"""
Add a custom InitStep category
==============================

By default :func:`execute_initialization` routes :class:`InitStep`
results into one of four ``Context`` slots: ``fields``, ``models``,
``operators``, ``resource`` (``mesh`` / ``runtime``). Unknown
categories fall through to ``models`` with a logged warning.

If you have a new family of objects to track (e.g. turbulence
closures, boundary patches, post-processors), register a handler on a
:class:`CategoryRouter` and pass it through.
"""

# %%
# Imports
# -------

from typing import Any

from neofoam.framework.context import Context
from neofoam.framework.initialization import InitStep
from neofoam.framework.initialization.execution import (
    default_router,
    execute_initialization,
)

# %%
# Register a handler
# ------------------
# A router maps category strings to ``(ctx, name, value) -> None``
# callables that write the produced value into ``ctx`` in whatever
# shape makes sense. Start from ``default_router()`` to inherit the
# four built-in handlers, then add your own.


def _route_turbulence(ctx: Context, name: str, value: Any) -> None:
    ctx.models[f"turbulence.{name}"] = value


router = default_router()
router.register("turbulence", _route_turbulence)


# %%
# Emit steps with the new category
# --------------------------------
# The ``field`` / ``operator`` / ``model`` / ``lazy`` factories in
# :mod:`~neofoam.framework.initialization.helpers` hard-code their
# category. For a custom category, construct :class:`InitStep`
# directly — ``category`` is a plain string, no closed enum, no
# ``Literal`` to extend.


class _KEpsilonModel:
    def __init__(self, U: Any) -> None:
        self.U = U


steps = [
    InitStep(
        name="fields.U",
        depends_on=[],
        initializer=lambda ctx: object(),  # stand-in for a real field
        category="fields",
    ),
    InitStep(
        name="turbulence.kepsilon",
        depends_on=["fields.U"],
        initializer=lambda ctx: _KEpsilonModel(ctx["fields.U"]),
        category="turbulence",
    ),
]

# %%
# Run with the custom router
# --------------------------
# Pass the configured router into :func:`execute_initialization`.
# Steps whose ``InitStep.category == "turbulence"`` are routed through
# ``_route_turbulence``; everything else uses the four built-in
# handlers plus the unknown-category fallback.

ctx = execute_initialization(steps, router=router)
print(sorted(ctx.models))

# %%
# When to use it
# --------------
#
# - A new step type has its own *meaningful* container in ``Context``
#   — use the router to write to that container directly.
# - You want a single warning per misrouted step instead of silent
#   fallback to ``models`` — register a handler that raises or logs.
# - Tests want to assert on category-specific routing in isolation —
#   register the handler, drive a couple of ``InitStep``\\ s through
#   :func:`build_context_from_results`, assert on the router's
#   observations.
#
# See also
# ~~~~~~~~
#
# - :doc:`/explanation/three-stage-init` — where the router sits in
#   the LOAD → RESOLVE → BUILD pipeline.
# - :doc:`/reference/initialization/execution/context_builder` —
#   ``CategoryRouter`` API.
