# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — see field_writer.py.

"""pybFoam backend wiring for the framework ``solutionLoop`` core Model.

The *model* — load ``TimeControlConfig`` from ``system/controlDict``, build the
pure-Python stepper + :class:`~neofoam.algorithms.solution_loop.SolutionLoop`
engine, and the loop-body operations (``set_time_step`` / ``increment_time``) plus
the outer predicate — lives in the framework
(:mod:`neofoam.algorithms.solution_loop`) and is re-exported here so the
solver entrypoint and ``create_fields`` keep importing the same names.

This module adds only the pybFoam backend touch-points the framework loop model
delegates to (DIP):

* :class:`PybFoamStepSink` — mirrors the Python stepper onto ``pybFoam.Time`` so
  field IO lands in the right time directory;
* :func:`loop_backend_steps` — init steps that inject the sink into the
  framework-built stepper and register the ``courant_provider`` (CFL measured on
  the live ``phi``) and ``loop_logger`` (``pybFoam.Info``) the framework
  operations consult.
"""

from typing import Any

from pybFoam import Info, computeCFLNumber

from neofoam.framework.initialization import InitStep, model
from neofoam.algorithms.solution_loop import (  # re-exported for the solver
    SolutionLoopPredicate,
    build,
    increment_time,
    make_solution_loop,
    make_stepper,
    set_time_step,
    solutionLoop,
)

__all__ = [
    "solutionLoop",
    "SolutionLoopPredicate",
    "build",
    "set_time_step",
    "increment_time",
    "make_stepper",
    "make_solution_loop",
    "PybFoamStepSink",
    "loop_backend_steps",
]


class PybFoamStepSink:
    """Mirror the Python stepper onto ``pybFoam.Time`` (a ``StepSink``).

    The stepper owns the advancement *logic* (deltaT, run, outputTime — all in
    Python); this keeps OpenFOAM's ``Time`` in lockstep so field IO lands in the
    right time directory. We push the (already-computed) deltaT and let
    ``Foam::Time`` take the matching step via ``increment`` — ``setTime`` can't be
    called from Python (``instant`` has no binding constructor). It does not
    write — that is the ``fieldWriter`` Model's job.
    """

    def __init__(self, runtime: Any) -> None:
        self._t = runtime

    def set_delta_t(self, dt: float) -> None:
        self._t.setDeltaT(dt)

    def advance_to(self, value: float, index: int) -> None:
        # deltaT was already pushed via set_delta_t; advance by the same step
        self._t.increment()


def loop_backend_steps() -> list[InitStep]:
    """Init steps wiring the pybFoam backend into the framework ``solutionLoop``.

    * injects :class:`PybFoamStepSink` into the framework-built stepper (which
      defaults to a no-op ``NullStepSink``);
    * registers ``courant_provider`` — CFL on the live ``phi``, pushed into the
      engine by ``set_time_step`` when stability constraints are present;
    * registers ``loop_logger`` — ``pybFoam.Info`` for the per-step time print.
    """

    def inject_sink(ctx: dict[str, Any]) -> PybFoamStepSink:
        sink = PybFoamStepSink(ctx["runtime"])
        ctx["models.stepper"].set_sink(sink)
        return sink

    def make_courant_provider(ctx: dict[str, Any]) -> Any:
        return lambda c: computeCFLNumber(c.fields["phi"])[0]

    def make_logger(ctx: dict[str, Any]) -> Any:
        return Info

    return [
        model("stepper_sink", inject_sink, depends_on=["models.stepper", "runtime"]),
        model("courant_provider", make_courant_provider),
        model("loop_logger", make_logger),
    ]
