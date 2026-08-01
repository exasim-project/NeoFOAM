# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — keeps parity with the framework
# model modules (the dependency_resolver matches ``param.annotation is Context``).

"""blockAMR backend wiring for the framework ``solutionLoop`` + ``fieldWriter``.

Both framework core Models are backend-agnostic; this module supplies the
block-structured touch-points through their injection seams (DIP):

* :class:`PlotfileWriteHook` — a ``FieldHook`` that writes an AMReX plotfile of
  the ``write=True`` fields on write steps (gated by the ``FieldWriter``).

The framework ``LoopState`` is the only clock: the projection operations read
``ctx.time`` for ``dt`` and the step time, so the loop needs no backend mirror
(the framework's ``NullLoopBackend`` default stands). The mesh and the loop state
are resolved from the init work dict at wire time and held on the hook — never
captured in an operation closure.
"""

from typing import Any, Literal, Mapping

from neofoam.algorithms.field_writer.field_writer import (  # re-exported for the solver
    build,
    fieldWriter,
    write_output,
)
from neofoam.algorithms.field_writer.writer import FieldHook
from neofoam.framework.initialization import InitStep, model

__all__ = [
    "fieldWriter",
    "build",
    "write_output",
    "PlotfileWriteHook",
    "loop_backend_steps",
    "writer_backend_steps",
]


@FieldHook.register
class PlotfileWriteHook(FieldHook):
    """Write an AMReX plotfile of the flagged fields on a write step.

    Receives only the ``write=True`` fields (``U`` / ``p``) — the same
    ``CellField`` objects registered in ``ctx.fields`` — and hands them to the
    ``blockamr`` free ``write_plotfile``, stamped with the loop's current
    simulation time. Plotfile directories are named ``plt<NNNNN>`` by write index
    (AMReX convention).
    """

    field_hook_type: Literal["plotfile"] = "plotfile"
    mesh: Any = None  # the blockamr mesh the fields live on
    loop_state: Any = None  # the framework LoopState (ctx.time)
    directory: str = "."
    count: int = 0

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        from blockamr.incompressible import (  # noqa: PLC0415 — lazy: keep import GPU-free
            write_plotfile,
        )

        to_write = [fields[n] for n in ("U", "p") if n in fields]
        name = f"{self.directory.rstrip('/')}/plt{self.count:05d}"
        write_plotfile(self.mesh, self.loop_state.value, name, to_write)
        self.count += 1


def loop_backend_steps() -> list[InitStep]:
    """Init steps wiring the blockAMR side of the framework ``solutionLoop``.

    Only ``loop_logger`` (a plain ``print`` for the per-step time announcement):
    the projection operations read the framework ``LoopState`` directly, so no
    ``LoopBackend`` mirror is needed.
    """

    def make_logger(ctx: dict[str, Any]) -> Any:
        return print

    return [model("loop_logger", make_logger)]


def writer_backend_steps(config: Any = None) -> list[InitStep]:
    """Init steps wiring the blockAMR backend into the framework ``fieldWriter``.

    Injects :class:`PlotfileWriteHook` into the framework-built ``FieldWriter``
    (which defaults to a no-op ``NullFieldHook``) and registers a no-op
    ``step_reporter`` (the framework ``write_output`` calls it after a write).
    """

    def inject_hook(ctx: dict[str, Any]) -> FieldHook:
        hook = PlotfileWriteHook(mesh=ctx["_blockamr_mesh"], loop_state=ctx["time"])
        ctx["models.writer"].hook = hook
        return hook

    def make_reporter(ctx: dict[str, Any]) -> Any:
        return lambda: None

    return [
        model(
            "writer_hook",
            inject_hook,
            depends_on=["models.writer", "_blockamr_mesh", "time"],
        ),
        model("step_reporter", make_reporter),
    ]
