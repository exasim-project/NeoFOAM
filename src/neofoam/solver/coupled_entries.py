# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``coupled`` entries of a solver's coupled patches, kept in every written field file.

The boundary defines the coupling: a patch of ``0/T`` is coupled when a
``coupled <condition>;`` entry names its condition. Neither OpenFOAM's
``fixedValue`` (``incompressibleFluid``) nor NeoN's ``calculated``
(``heatConductionNeoN``) writes that entry back, so a run restarted from a
written time would be uncoupled. :func:`keep_coupled_entries` wraps a
``FieldWriter``'s hook in a :class:`CoupledEntriesWriteHook`, which adds the
entries back to each file after the inner hook wrote it.
"""

import gzip
import re
from pathlib import Path
from typing import Any, Literal, Mapping

import pybFoam as pyf

from neofoam.algorithms.field_writer.writer import FieldHook

__all__ = ["CoupledEntriesWriteHook", "keep_coupled_entries", "restore_coupled_entries"]


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

    The written patch (OpenFOAM's ``fixedValue``, NeoN's ``calculated``) has no
    ``coupled`` entry, so the patches the boundary couples (``coupled
    <condition>;`` in ``0/T``) would restart from a written time uncoupled.
    ``entries`` maps each field to its coupled patches and their conditions;
    the field files are those of the case directory ``case_dir``
    (``processor<N>/`` on more than one process, as OpenFOAM's ``argList`` sets
    the case) at the time of ``time`` (the ``Foam::Time`` the inner hook writes
    at: its ``timeName()``).
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
