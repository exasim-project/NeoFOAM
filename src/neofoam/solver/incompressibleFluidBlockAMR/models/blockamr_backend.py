# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — keeps parity with the framework
# model modules (the dependency_resolver matches ``param.annotation is Context``).

"""blockAMR backend wiring for the framework ``solutionLoop`` + ``fieldWriter``.

Both framework core Models are backend-agnostic; this module supplies the two
block-structured touch-points through their injection seams (DIP):

* :class:`BlockAMRTimeBackend` — a ``LoopBackend`` mirroring the framework
  ``LoopState`` clock onto the projection state (which owns its own ``dt``/time).
  The framework loop owns the outer step count + write control; this backend only
  pushes the per-step ``dt`` onto the state. It deliberately does **not** write
  the simulation time — the ``project`` operation advances that itself, so
  mirroring it here would double-count.
* :class:`PlotfileWriteHook` — a ``FieldHook`` that writes an AMReX plotfile of
  the ``write=True`` fields on write steps (gated by the ``FieldWriter``).

The projection state is resolved from the init work dict (``_projection_state``)
at wire time and held on the backend/hook — never captured in an operation
closure.
"""

from typing import Any, Literal, Mapping

from neofoam.algorithms.field_writer.field_writer import (  # re-exported for the solver
    build,
    fieldWriter,
    write_output,
)
from neofoam.algorithms.field_writer.writer import FieldHook
from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.framework.initialization import InitStep, model

__all__ = [
    "fieldWriter",
    "build",
    "write_output",
    "BlockAMRTimeBackend",
    "PlotfileWriteHook",
    "loop_backend_steps",
    "writer_backend_steps",
]


class BlockAMRTimeBackend:
    """LoopBackend: mirror the framework ``LoopState`` ``dt`` onto the state.

    The projection state owns its own ``dt`` and simulation time (the ``project``
    operation does ``state.t += dt``). The framework loop drives the outer step
    count and write control; each ``update`` pushes the (possibly
    constraint-folded) ``delta_t`` onto the state so the next ``project`` uses
    it. Simulation time is left to the operation to avoid double-advancing it.
    """

    def __init__(self, state: Any) -> None:
        self._state = state

    def update(self, state: LoopState) -> None:
        self._state.dt = state.delta_t


@FieldHook.register
class PlotfileWriteHook(FieldHook):
    """Write an AMReX plotfile of the flagged fields on a write step.

    Receives only the ``write=True`` fields (``U`` / ``p``) — the same
    ``CellField`` objects the projection state owns — and hands them to the
    ``blockamr`` free ``write_plotfile``, stamped with the state's current
    simulation time. Plotfile directories are named ``plt<NNNNN>`` by write index
    (AMReX convention).
    """

    field_hook_type: Literal["plotfile"] = "plotfile"
    state: Any = None  # the projection state (fields + equations + time)
    directory: str = "."
    count: int = 0

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        from blockamr.incompressible import write_plotfile

        to_write = [fields[n] for n in ("U", "p") if n in fields] or [self.state.U]
        name = f"{self.directory.rstrip('/')}/plt{self.count:05d}"
        write_plotfile(self.state.mesh, self.state.time, name, to_write)
        self.count += 1


def loop_backend_steps() -> list[InitStep]:
    """Init steps wiring the blockAMR backend into the framework ``solutionLoop``.

    Injects :class:`BlockAMRTimeBackend` into the framework loop (which defaults
    to a no-op ``NullLoopBackend``); ``set_backend`` immediately mirrors the
    initial state, seeding the projection state's ``dt``. Registers
    ``loop_logger`` as a plain ``print`` for the per-step time announcement.
    """

    def inject_backend(ctx: dict[str, Any]) -> BlockAMRTimeBackend:
        backend = BlockAMRTimeBackend(ctx["_projection_state"])
        ctx["models.solution_loop"].set_backend(backend)
        return backend

    def make_logger(ctx: dict[str, Any]) -> Any:
        return print

    return [
        model(
            "loop_backend",
            inject_backend,
            depends_on=["models.solution_loop", "_projection_state"],
        ),
        model("loop_logger", make_logger),
    ]


def writer_backend_steps(config: Any = None) -> list[InitStep]:
    """Init steps wiring the blockAMR backend into the framework ``fieldWriter``.

    Injects :class:`PlotfileWriteHook` into the framework-built ``FieldWriter``
    (which defaults to a no-op ``NullFieldHook``) and registers a no-op
    ``step_reporter`` (the framework ``write_output`` calls it after a write).
    """

    def inject_hook(ctx: dict[str, Any]) -> FieldHook:
        hook = PlotfileWriteHook(state=ctx["_projection_state"])
        ctx["models.writer"].hook = hook
        return hook

    def make_reporter(ctx: dict[str, Any]) -> Any:
        return lambda: None

    return [
        model(
            "writer_hook",
            inject_hook,
            depends_on=["models.writer", "_projection_state"],
        ),
        model("step_reporter", make_reporter),
    ]
