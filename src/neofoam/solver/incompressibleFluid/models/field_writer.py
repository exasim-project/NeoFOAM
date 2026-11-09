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
  (``step_reporter``) the framework ``write_output`` consults;
* :class:`CoupledEntriesWriteHook` (:func:`keep_coupled_entries`) — after another
  hook wrote, adds the ``coupled`` entries of the coupled patches back, which
  OpenFOAM's ``fixedValue`` drops on write, so a written time restarts coupled.
"""

import gzip
import re
from pathlib import Path
from typing import Any, Literal, Mapping

import pybFoam as pyf

from neofoam.algorithms.field_writer.field_writer import (  # re-exported for the solver
    build,
    fieldWriter,
    write_output,
)
from neofoam.algorithms.field_writer.writer import FieldHook
from neofoam.framework.initialization import InitStep, model

from ..configs import ControlDictConfig

__all__ = [
    "fieldWriter",
    "build",
    "write_output",
    "RuntimeWriteHook",
    "PerFieldWriteHook",
    "make_field_hook",
    "writer_backend_steps",
    "CoupledEntriesWriteHook",
    "keep_coupled_entries",
    "restore_coupled_entries",
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


def restore_coupled_entries(path: Path, entries: Mapping[str, str]) -> None:
    """Add a ``coupled <condition>;`` entry to each patch of ``entries`` in field file ``path``.

    ``path`` as OpenFOAM writes it (``path.gz`` if compressed; ascii or binary):
    each patch of ``boundaryField`` opens with its name and a brace on lines of
    their own, at four spaces; the entry goes first in its block. A patch that
    has the entry already is left alone. Raises ``ValueError`` when the file
    holds no such patch.
    """
    compressed = not path.is_file() and path.with_name(path.name + ".gz").is_file()
    target = path.with_name(path.name + ".gz") if compressed else path
    data = gzip.decompress(target.read_bytes()) if compressed else target.read_bytes()
    boundary = data.find(b"boundaryField")
    if boundary < 0:
        raise ValueError(f"{target}: no boundaryField to add the coupled entries to")
    for patch, condition in entries.items():
        opening = re.compile(rb"\n    " + re.escape(patch.encode()) + rb"[ \t]*\n    \{[ \t]*\n")
        found = opening.search(data, boundary)
        if found is None:
            raise ValueError(
                f"{target}: no patch {patch} in boundaryField to add 'coupled {condition}' to"
            )
        head = data[found.end() : found.end() + 4096].split(b"\n    }", 1)[0]
        if re.search(rb"^\s*coupled\s", head.split(b"value", 1)[0], re.MULTILINE):
            continue
        entry = f"        coupled         {condition};\n".encode()
        data = data[: found.end()] + entry + data[found.end() :]
    target.write_bytes(gzip.compress(data) if compressed else data)


@FieldHook.register
class CoupledEntriesWriteHook(FieldHook):
    """Write through ``inner``, then add the ``coupled`` entries back (a :class:`FieldHook`).

    OpenFOAM's ``fixedValue`` writes no ``coupled`` entry, so the patches the
    boundary couples (``coupled <condition>;`` in ``0/T``) would restart from a
    written time uncoupled. ``entries`` maps each field to its coupled patches and
    their conditions; the field files are those of the case directory
    ``case_dir`` (``processor<N>/`` on more than one process, as OpenFOAM's
    ``argList`` sets the case) at the time of ``time``.
    """

    field_hook_type: Literal["coupledEntries"] = "coupledEntries"
    inner: Any = None
    time: Any = None
    case_dir: Path = Path(".")
    entries: dict[str, dict[str, str]] = {}

    def time_directory(self) -> Path:
        """The directory OpenFOAM writes the fields of the current time to."""
        case = self.case_dir
        if pyf.Pstream.parRun() and pyf.Pstream.nProcs() > 1:
            case = case / f"processor{pyf.Pstream.myProcNo()}"
        return case / str(self.time.timeName())

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        self.inner.write_fields(fields)
        directory = self.time_directory()
        for field, patches in self.entries.items():
            restore_coupled_entries(directory / field, patches)


def keep_coupled_entries(
    writer: Any, time: Any, case_dir: Path, entries: Mapping[str, Mapping[str, str]]
) -> None:
    """Make ``writer`` (a ``FieldWriter``) add the ``coupled`` entries back after each write.

    ``entries``: field -> patch -> condition. Nothing changes without one.
    """
    if not any(entries.values()):
        return
    writer.hook = CoupledEntriesWriteHook(
        inner=writer.hook,
        time=time,
        case_dir=Path(case_dir),
        entries={field: dict(patches) for field, patches in entries.items() if patches},
    )
