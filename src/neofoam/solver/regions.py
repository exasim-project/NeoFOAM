# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What the solvers share to run as the coupled regions of a ``multiRun`` case."""

from pathlib import Path
from typing import Any, Mapping, Optional

import pybFoam as pyf
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.framework.coupling import Array, FaceData
from neofoam.io import YAML, IOStrategy

__all__ = [
    "ControlDictConfig",
    "RegionSolverConfig",
    "WallHeatFlux",
    "WallTemperature",
    "coupled_entries",
    "region_case_dir",
]


class RegionSolverConfig(BaseModel):
    """The solver of one region and the MPI ``processes`` it runs on.

    In the yaml a plain solver name stands for one process. ``processes`` counts
    only under ``mpirun`` (:mod:`~neofoam.solver.multiRun.mpi`); without MPI
    every region runs on a thread of its own, whatever it says.
    """

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
    participants: dict[str, RegionSolverConfig] = Field(
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
