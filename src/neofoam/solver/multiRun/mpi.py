# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun under ``mpirun``: MPI_COMM_WORLD split into the regions of ``regionSolvers``.

The regions take the ranks in blocks of their ``processes``, in stepping order
(:class:`RegionLayout`); that is OpenFOAM's split under ``-parallel -world
<region>``, which every region runs, also one on one process (``processor0``).
The split is collective over every process, so every region takes part.
mpi4py (the extra ``neofoam[mpi]``) is imported here only, under MPI only.
"""

import atexit
import gc
import os
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# the environment variables in which MPI launchers pass the number of processes
SIZE_VARIABLES = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE")


class MpiSetupError(ValueError):
    """The case cannot run on the MPI processes it was started on."""


def launched_size() -> int:
    """The number of MPI processes the launcher started, from its environment; 1 without one."""
    for variable in SIZE_VARIABLES:
        value = os.environ.get(variable)
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return 1


def under_mpi() -> bool:
    """Whether multiRun was started by an MPI launcher on more than one process."""
    return launched_size() > 1


def import_mpi() -> Any:
    """mpi4py's ``MPI`` (it initializes MPI); raises ``ImportError`` naming the extra without it."""
    try:
        from mpi4py import MPI  # noqa: PLC0415 - the optional extra neofoam[mpi], MPI runs only
    except ImportError as error:
        raise ImportError(
            f"the case was started on {launched_size()} MPI processes, but mpi4py is not "
            "installed; install the extra neofoam[mpi] (mpi4py built against OpenFOAM's MPI)"
        ) from error
    return MPI


@dataclass(frozen=True)
class RegionLayout:
    """The regions on the ranks of MPI_COMM_WORLD: blocks of ``processes``, in stepping order."""

    processes: Mapping[str, int]

    @property
    def total(self) -> int:
        """The number of processes of all regions."""
        return sum(self.processes.values())

    def index(self, region: str) -> int:
        """The position of ``region`` in the stepping order (its MPI ``Split`` colour)."""
        return list(self.processes).index(region)

    def first(self, region: str) -> int:
        """The first global rank of ``region``."""
        regions = list(self.processes)
        return sum(self.processes[other] for other in regions[: regions.index(region)])

    def ranks(self, region: str) -> range:
        """The global ranks of ``region``."""
        first = self.first(region)
        return range(first, first + self.processes[region])

    def region_of(self, rank: int) -> str:
        """The region global rank ``rank`` runs."""
        for region in self.processes:
            if rank in self.ranks(region):
                return region
        raise MpiSetupError(f"rank {rank} runs no region of {self}")

    def argv(self, region: str) -> list[str]:
        """The arguments of ``region``'s solver: ``-parallel -world <region>``, on any processes."""
        return ["multiRun", "-parallel", "-world", region]

    def __str__(self) -> str:
        parts = []
        for region in self.processes:
            ranks = self.ranks(region)
            where = f"rank {ranks[0]}" if len(ranks) == 1 else f"ranks {ranks[0]}-{ranks[-1]}"
            parts.append(f"{region}: {where}")
        return ", ".join(parts)


def check_size(layout: RegionLayout, size: int) -> None:
    """Raise ``MpiSetupError`` unless ``size``, MPI_COMM_WORLD's, is the regions' processes."""
    if size != layout.total:
        counts = " + ".join(f"{count} ({region})" for region, count in layout.processes.items())
        raise MpiSetupError(
            f"the case was started on {size} MPI processes, but the regions of "
            f"system/controlDict.yaml run on {counts} = {layout.total}; start it with "
            f"mpirun -np {layout.total}, or change the regions' processes"
        )


def check_decomposed(layout: RegionLayout, case_dir: Path = Path(".")) -> None:
    """Raise ``MpiSetupError`` unless a region on n processes has exactly ``processor0..n-1``."""
    problems = []
    for region, count in layout.processes.items():
        directory = case_dir / region
        on = f"{count} process" if count == 1 else f"{count} processes"
        missing = [
            f"processor{n}" for n in range(count) if not (directory / f"processor{n}").is_dir()
        ]
        if missing:
            problems.append(
                f"region {region} runs on {on}, but {region}/ lacks "
                f"{', '.join(missing)}; decompose it into {count} (numberOfSubdomains "
                f"{count} in {region}/system/decomposeParDict, decomposePar -case {region})"
            )
        elif (directory / f"processor{count}").is_dir():
            problems.append(
                f"region {region} runs on {on}, but {region}/ is decomposed "
                f"into more ({region}/processor{count} exists); decompose it into {count}"
            )
    if problems:
        raise MpiSetupError("; ".join(problems))


def check_world(region: str, comm: Any) -> None:
    """Raise ``MpiSetupError`` unless OpenFOAM's and NeoN's world of ``region`` is ``comm``."""
    pybfoam = sys.modules.get("pybFoam")
    if pybfoam is not None and pybfoam.Pstream.parRun():
        world = (int(pybfoam.Pstream.myProcNo()), int(pybfoam.Pstream.nProcs()))
        if world != (comm.rank, comm.size):
            raise MpiSetupError(
                f"region {region}: OpenFOAM's world is rank {world[0]} of {world[1]}, its MPI "
                f"split rank {comm.rank} of {comm.size}"
            )
    neon = sys.modules.get("neon._neon")
    if neon is not None:
        default = type(comm).f2py(neon.default_communicator())
        if (default.rank, default.size) != (comm.rank, comm.size):
            raise MpiSetupError(
                f"region {region}: NeoN's communicator is rank {default.rank} of "
                f"{default.size}, its MPI split rank {comm.rank} of {comm.size}"
            )


def abort(world: Any, message: str) -> None:
    """Print ``message`` with the rank, flush, and abort every process of the run."""
    sys.stdout.flush()
    print(f"[multiRun rank {world.rank}/{world.size}] {message}", file=sys.stderr, flush=True)
    world.Abort(1)


def abort_all(world: Any, message: str) -> None:
    """Abort a run every process has found unfit: rank 0 prints ``message`` first."""
    if world.rank == 0:
        print(f"multiRun: {message}", file=sys.stderr, flush=True)
    world.Barrier()
    world.Abort(1)


def failure(error: BaseException) -> str:
    """``error`` as reported: its message for an ``MpiSetupError``, else its traceback."""
    if isinstance(error, MpiSetupError):
        return str(error)
    return "".join(traceback.format_exception(error))


# what must outlive the run: an OpenFOAM argList finalizes MPI as it is destroyed
_ALIVE: list[Any] = []


def keep_alive(mpi: Any, value: Any) -> None:
    """Keep ``value`` until ``mpi`` is finalized at exit, then release it.

    An OpenFOAM ``argList`` finalizes MPI in its destructor unless MPI is
    finalized already; NeoN fields must go before Kokkos is finalized (an
    earlier exit handler, so it runs later).
    """
    if not _ALIVE:
        atexit.register(_finalize, mpi)
    _ALIVE.append(value)


def _finalize(mpi: Any) -> None:
    if not mpi.Is_finalized():
        mpi.COMM_WORLD.Barrier()
        mpi.Finalize()
    _ALIVE.clear()
    gc.collect()
