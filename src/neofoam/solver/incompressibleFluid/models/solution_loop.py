# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — see field_writer.py.

"""pybFoam backend wiring for the framework ``solutionLoop`` core Model.

The *model* — load ``TimeControlConfig`` from ``system/controlDict``, build the
``LoopState`` (``ctx.time``) + the
:class:`~neofoam.algorithms.solution_loop.solution_loop.SolutionLoop` engine, and
the loop-body operations (``set_time_step`` / ``increment_time``) plus the outer
predicate — lives in the framework
(:mod:`neofoam.algorithms.solution_loop.solution_loop`) and is re-exported here so
the solver entrypoint and ``create_fields`` keep importing the same names.

This module adds only the pybFoam-specific touch-points the framework loop model
delegates to (DIP). The :class:`FoamTime` ``LoopBackend`` itself lives in the
backend-agnostic ``algorithms`` package (it is duck-typed on the wrapped time
object); here we only:

* :func:`loop_backend_steps` — init steps that construct the ``FoamTime`` backend
  from the injected ``pybFoam.Time`` and inject it into the framework-built engine,
  and register the ``loop_logger`` (``pybFoam.Info``) the framework operations
  consult. The CFL ``measurement_provider.courant`` is no longer registered here —
  it belongs to the opt-in ``adaptiveTimeStep`` model, so the measurement exists
  only when adaptive stepping is selected.
"""

from typing import Any

from pybFoam import Info

from neofoam.algorithms.solution_loop.foam_time import FoamTime  # re-exported
from neofoam.algorithms.solution_loop.solution_loop import (  # re-exported for the solver
    SolutionLoopPredicate,
    build,
    increment_time,
    make_loop_state,
    make_solution_loop,
    set_time_step,
    solutionLoop,
)
from neofoam.framework.initialization import InitStep, model

__all__ = [
    "solutionLoop",
    "SolutionLoopPredicate",
    "build",
    "set_time_step",
    "increment_time",
    "make_loop_state",
    "make_solution_loop",
    "FoamTime",
    "loop_backend_steps",
]


def loop_backend_steps() -> list[InitStep]:
    """Init steps wiring the pybFoam backend into the framework ``solutionLoop``.

    * injects the :class:`FoamTime` backend into the framework-built engine
      (which defaults to a no-op ``NullLoopBackend``);
    * registers ``loop_logger`` — ``pybFoam.Info`` for the per-step time print.
    """

    def inject_backend(ctx: dict[str, Any]) -> FoamTime:
        backend = FoamTime(ctx["_foam_time"])
        ctx["models.solution_loop"].set_backend(backend)
        return backend

    def make_logger(ctx: dict[str, Any]) -> Any:
        return Info

    return [
        model(
            "loop_backend",
            inject_backend,
            depends_on=["models.solution_loop", "_foam_time"],
        ),
        model("loop_logger", make_logger),
    ]
