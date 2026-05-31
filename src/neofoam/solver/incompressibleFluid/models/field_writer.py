# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — keeps parity with the framework
# model modules (the dependency_resolver matches ``param.annotation is Context``).

"""pybFoam backend wiring for the framework ``fieldWriter`` core Model.

The *model* itself — load ``WriteControlConfig`` from ``system/controlDict``,
build a :class:`~neofoam.algorithms.field_writer.writer.FieldWriter`, and the per-step
``write_output`` operation — lives in the framework
(:mod:`neofoam.algorithms.field_writer`) and is re-exported here so the
solver entrypoint and ``create_fields`` keep importing the same names.

This module adds only what is backend-specific (DIP):

* the two :class:`~neofoam.algorithms.field_writer.writer.FieldHook` backends
  (:class:`RuntimeWriteHook` registry write, :class:`PerFieldWriteHook`), selected
  from ``controlDict`` ``writeBackend`` by :func:`make_field_hook`;
* :func:`writer_backend_steps` — the init steps that inject the selected hook into
  the framework-built ``FieldWriter`` and register the per-step timing reporter
  (``step_reporter``) the framework ``write_output`` consults.
"""

from typing import Any, Literal, Mapping

import pybFoam as pyf

from neofoam.framework.initialization import InitStep, model
from neofoam.algorithms.field_writer.field_writer import (  # re-exported for the solver
    build,
    fieldWriter,
    write_output,
)
from neofoam.algorithms.field_writer.writer import FieldHook

from ..configs import ControlDictConfig

__all__ = [
    "fieldWriter",
    "build",
    "write_output",
    "RuntimeWriteHook",
    "PerFieldWriteHook",
    "make_field_hook",
    "writer_backend_steps",
]


@FieldHook.register
class RuntimeWriteHook(FieldHook):
    """Persist via the mesh/time *registry* — ``pybFoam.Time.write`` (a :class:`FieldHook`).

    The OpenFOAM-faithful write: one call persists every registered field at the
    current (loop-synced) time, including sub-model fields not present in
    ``ctx.fields``. The ``fields`` argument is ignored (the registry knows them).
    The live ``Foam::Time`` registry is held in an ``Any`` field (injected at
    build, not from the file).
    """

    field_hook_type: Literal["runtime"] = "runtime"
    registry: Any = None

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        self.registry.write(True)


@FieldHook.register
class PerFieldWriteHook(FieldHook):
    """Persist each given field individually via ``pybFoam.write`` (a :class:`FieldHook`).

    Receives only the fields flagged ``write=True`` (via ``ctx.write_fields``), so
    transient intermediates (``UEqn``, …) are never handed in. The stepper has
    already synced ``Foam::Time``, so each field lands in the right time directory.
    """

    field_hook_type: Literal["perField"] = "perField"

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        for fld in fields.values():
            pyf.write(fld)


def make_field_hook(config: ControlDictConfig, registry: Any) -> FieldHook:
    """Select the write backend from ``config.writeBackend``."""
    if config.writeBackend == "perField":
        return PerFieldWriteHook()
    return RuntimeWriteHook(registry=registry)


def writer_backend_steps(config: ControlDictConfig) -> list[InitStep]:
    """Init steps wiring the pybFoam backend into the framework ``fieldWriter``.

    * injects the config-selected :class:`FieldHook` into the framework-built
      ``FieldWriter`` (which defaults to a no-op ``NullFieldHook``);
    * registers ``step_reporter`` — the per-step timing report the framework
      ``write_output`` calls after a write (``pybFoam.Time.printExecutionTime``).
    """

    def inject_hook(ctx: dict[str, Any]) -> FieldHook:
        hook = make_field_hook(config, ctx["_foam_time"])
        ctx["models.writer"].hook = hook
        return hook

    def make_reporter(ctx: dict[str, Any]) -> Any:
        registry = ctx["_foam_time"]
        return lambda: registry.printExecutionTime()

    return [
        model("writer_hook", inject_hook, depends_on=["models.writer", "_foam_time"]),
        model("step_reporter", make_reporter, depends_on=["_foam_time"]),
    ]
