# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What the solvers share to run as the coupled regions of a ``multiRun`` case."""

import gzip
import re
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import pybFoam as pyf
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from neofoam.algorithms.field_writer.writer import FieldHook
from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.framework.coupling import Array, FaceData
from neofoam.io import YAML, IOStrategy

__all__ = [
    "ControlDictConfig",
    "CoupledEntriesWriteHook",
    "ParticipantConfig",
    "WallHeatFlux",
    "WallTemperature",
    "coupled_entries",
    "keep_coupled_entries",
    "region_case_dir",
    "restore_coupled_entries",
]


class ParticipantConfig(BaseModel):
    """The solver of one region and its MPI ``processes``, used under ``mpirun`` only."""

    model_config = ConfigDict(extra="forbid")

    solver: str
    processes: PositiveInt = 1


@IOStrategy(YAML("system/controlDict.yaml"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict.yaml``: the time control of every region, and its solvers.

    ``participants`` maps each region to its solver, in stepping order; a region
    reads its own files under ``<region>/``. An entry is the solver's name or
    ``{solver: <name>, processes: <n>}`` (MPI processes under ``mpirun``).

    Example::

        endTime: 10.0
        deltaT: 1.0
        participants:
          fluid: {solver: myFluidSolver, processes: 2}
          solid: mySolidSolver
    """

    # The i18n prefix labels the wizard's "add a region" row (neofoam.ui.form_schema).
    participants: dict[str, ParticipantConfig] = Field(
        min_length=1, json_schema_extra={"i18n": "nf.region"}
    )

    @field_validator("participants", mode="before")
    @classmethod
    def _plain_names(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                region: {"solver": entry} if isinstance(entry, str) else entry
                for region, entry in value.items()
            }
        return value

    def solvers(self) -> dict[str, str]:
        """Every region's solver name, in stepping order."""
        return {region: entry.solver for region, entry in self.participants.items()}

    def processes(self) -> dict[str, int]:
        """Every region's number of MPI processes, in stepping order."""
        return {region: entry.processes for region, entry in self.participants.items()}


def region_case_dir(region: Optional[str]) -> Path:
    """The case directory of ``region``: ``<region>/``, or ``.`` without a region.

    NeoFOAM reads only the default mesh region, so each region of a multiRun
    case is an OpenFOAM case of its own.
    """
    return Path(region) if region is not None else Path(".")


class WallTemperature(FaceData):
    """What a Neumann side sends: its wall temperature per face."""

    T: Array


class WallHeatFlux(FaceData):
    """What a Dirichlet side sends: the heat flux per face area into its region."""

    q: Array


def coupled_entries(path: Path, patch_types: Mapping[str, str]) -> dict[str, str]:
    """The coupled patches of the field file ``path``: patch -> its ``coupled`` condition.

    ``patch_types`` maps each known condition to the ``type`` its patch must have;
    another condition or type raises ``ValueError``.

    Example::

        coupled_entries(Path("0/T"), {"coupledTemperature": "calculated"})
    """
    boundary = pyf.dictionary.read(str(path)).subDict("boundaryField")
    entries: dict[str, str] = {}
    for patch in map(str, boundary.toc()):
        entry = boundary.subDict(patch)
        if not entry.found("coupled"):
            continue
        condition = str(entry.get_word("coupled"))
        if condition not in patch_types:
            raise ValueError(
                f"{path}: patch {patch} is coupled as {condition!r}; known: {sorted(patch_types)}"
            )
        patch_type = str(entry.get_word("type"))
        if patch_type != patch_types[condition]:
            raise ValueError(
                f"{path}: the coupled patch {patch} has type {patch_type}; "
                f"a {condition} patch is {patch_types[condition]!r}"
            )
        entries[patch] = condition
    return entries


def restore_coupled_entries(path: Path, entries: Mapping[str, str]) -> None:
    """Add ``coupled <condition>;`` to each patch of ``entries`` in the written field file ``path``.

    For a file as OpenFOAM writes it (``path.gz`` if compressed): the entry goes
    first in the patch's block; a patch that has one is left alone. Raises
    ``ValueError`` when the file has no such patch.
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
    """Write through ``inner``, then add the ``coupled`` entries back to the written files.

    Neither OpenFOAM's ``fixedValue`` nor NeoN's ``calculated`` writes the entry, so
    a written time would restart uncoupled. Build it with :func:`keep_coupled_entries`.
    """

    field_hook_type: Literal["coupledEntries"] = "coupledEntries"
    inner: Any
    time: Any
    case_dir: Path
    entries: dict[str, dict[str, str]]

    def time_directory(self) -> Path:
        """The directory of the current time, ``processor<N>/`` on more than one process."""
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

    Use it in a coupled solver's init with what :func:`coupled_entries` read;
    nothing changes when no patch is coupled.

    Example::

        keep_coupled_entries(ctx.models["writer"], foam_time, Path("solid"), {"T": entries})
    """
    if not any(entries.values()):
        return
    writer.hook = CoupledEntriesWriteHook(
        inner=writer.hook,
        time=time,
        case_dir=Path(case_dir),
        entries={field: dict(patches) for field, patches in entries.items() if patches},
    )
