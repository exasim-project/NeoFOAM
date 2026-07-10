# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — see field_writer.py.

"""NeoN backend wiring for the framework ``solutionLoop`` core Model.

The *model* — load ``TimeControlConfig``, build the ``LoopState`` (``ctx.time``)
+ the :class:`~neofoam.algorithms.solution_loop.solution_loop.SolutionLoop`
engine, and the loop-body operations plus the outer predicate — lives in the
framework and is re-exported here so the solver entrypoint and
``create_fields`` keep importing the same names.

This module adds only the NeoN-specific touch-point: :class:`NeoNTimeSync`, a
composite ``LoopBackend`` replacing the legacy ``nfb.sync_run_times``. That C++
helper did two things per step: (a) the CFL ``setDeltaT`` adjust — now owned by
the ``timeStepConstraint`` fold (the ``courant`` / ``maxDeltaT`` contribution
models) — and (b) mirroring ``dt`` / ``t`` onto the NeoN ``RunTime``, whose
``dt``/``t`` are plain cached members that nothing reads live from
``Foam::Time``. (b) is this backend: it wraps the framework
:class:`~neofoam.algorithms.solution_loop.foam_time.FoamTime` mirror (so
``Foam::Time`` stays in lockstep for IO time directories) and pushes the
advanced state onto the adapter on every ``update()``.
"""

from typing import Any

from neofoam.algorithms.solution_loop.foam_time import FoamTime
from neofoam.algorithms.solution_loop.loop_state import LoopState
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
    "NeoNTimeSync",
    "neon_loop_backend_steps",
]


class NeoNTimeSync:
    """LoopBackend: mirror the LoopState onto ``Foam::Time`` AND the NeoN RunTime.

    Holds the pybFoam ``argList`` + ``Time`` references for the whole run —
    ``Foam::Time`` keeps a raw reference to the argList and the
    ``create_adapter_run_time`` binding keeps no reference to the Time, so the
    ``MeshAdapter`` inside the NeoN ``RunTime`` must never outlive either.
    """

    def __init__(self, foam_time: Any, neon_runtime: Any, arg_list: Any) -> None:
        self._arg_list = arg_list
        self._foam_time = foam_time
        self._foam = FoamTime(foam_time)
        self._rt = neon_runtime

    def update(self, state: LoopState) -> None:
        self._foam.update(state)
        self._rt.dt = state.delta_t
        self._rt.t = state.value


def neon_loop_backend_steps() -> list[InitStep]:
    """Init steps wiring the NeoN backend into the framework ``solutionLoop``.

    * injects the :class:`NeoNTimeSync` backend into the framework-built engine
      (which defaults to a no-op ``NullLoopBackend``); ``set_backend``
      immediately mirrors the initial state, seeding ``rt.dt`` / ``rt.t``;
    * registers ``loop_logger`` — plain ``print``, matching the legacy port's
      Python-side time announcement.
    """

    def inject_backend(ctx: dict[str, Any]) -> NeoNTimeSync:
        backend = NeoNTimeSync(
            ctx["_foam_time"], ctx["_neon_runtime"], ctx["_arg_list"]
        )
        ctx["models.solution_loop"].set_backend(backend)
        return backend

    def make_logger(ctx: dict[str, Any]) -> Any:
        return print

    return [
        model(
            "loop_backend",
            inject_backend,
            depends_on=[
                "models.solution_loop",
                "_arg_list",
                "_foam_time",
                "_neon_runtime",
            ],
        ),
        model("loop_logger", make_logger),
    ]
