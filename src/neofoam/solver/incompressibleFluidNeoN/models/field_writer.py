# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — keeps parity with the framework
# model modules (the dependency_resolver matches ``param.annotation is Context``).

"""NeoN backend wiring for the framework ``fieldWriter`` core Model.

The *model* itself — load ``WriteControlConfig``, build a ``FieldWriter``, and
the per-step ``write_output`` operation — lives in the framework and is
re-exported here so the solver entrypoint and ``create_fields`` keep importing
the same names.

This module adds only what is NeoN-specific (DIP): NeoN fields live in the
NeoN ``Database``, not the OpenFOAM objectRegistry, so the pybFoam registry
write (``Time.write(True)``) cannot see them. :class:`NeoNWriteHook` writes
the flagged fields through ``nfb.write_scalar_field`` / ``write_vector_field``
and lets the turbulence model persist its own fields (``nut`` / ``nuTilda``
for SA-DDES), exactly like the legacy port's ``outputTime()`` branch.
"""

from typing import Any, Literal, Mapping

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings

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
    "NeoNWriteHook",
    "neon_writer_backend_steps",
]


@FieldHook.register
class NeoNWriteHook(FieldHook):
    """Persist NeoN fields via the NeoN writers (a :class:`FieldHook`).

    Receives only the fields flagged ``write=True`` (``p`` / ``U``), dispatched
    by name to the scalar/vector writer; the turbulence model rides along so
    its own fields land on disk too. The ``NeoNTimeSync`` backend has already
    synced ``Foam::Time``, so everything lands in the right time directory.
    """

    field_hook_type: Literal["neon"] = "neon"
    runtime: Any = None  # the NeoN RunTime adapter
    turbulence: Any = None  # the turbulence handle (NeoNHandle)
    scalar_names: tuple[str, ...] = ("p",)
    vector_names: tuple[str, ...] = ("U",)

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        print("Writing fields")
        for name, fld in fields.items():
            if name in self.scalar_names:
                nfb.write_scalar_field(fld, self.runtime)
            elif name in self.vector_names:
                nfb.write_vector_field(fld, self.runtime)
        self.turbulence.write(self.runtime.mesh)


def neon_writer_backend_steps() -> list[InitStep]:
    """Init steps wiring the NeoN backend into the framework ``fieldWriter``.

    * injects the :class:`NeoNWriteHook` into the framework-built
      ``FieldWriter`` (which defaults to a no-op ``NullFieldHook``);
    * registers ``step_reporter`` — the per-step timing report the framework
      ``write_output`` calls (``pybFoam.Time.printExecutionTime``, matching the
      legacy port's per-step print).
    """

    def inject_hook(ctx: dict[str, Any]) -> FieldHook:
        hook = NeoNWriteHook(
            runtime=ctx["_neon_runtime"], turbulence=ctx["models.turbulence"]
        )
        ctx["models.writer"].hook = hook
        return hook

    def make_reporter(ctx: dict[str, Any]) -> Any:
        registry = ctx["_foam_time"]
        return lambda: registry.printExecutionTime()

    return [
        model(
            "writer_hook",
            inject_hook,
            depends_on=["models.writer", "models.turbulence", "_neon_runtime"],
        ),
        model("step_reporter", make_reporter, depends_on=["_foam_time"]),
    ]
