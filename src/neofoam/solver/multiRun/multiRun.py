# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun: one solver per region in one case, each on its own thread.

``system/controlDict.yaml`` lists the region solvers; each region is an ordinary
solver instantiated with ``region=``. Regions share nothing but a
:class:`~neofoam.framework.coupling.CommChannel`, over which the coupled patches
listed in ``system/coupling.yaml`` exchange their data. Under ``mpirun``
each region runs on processes of its own (:mod:`~neofoam.solver.multiRun.mpi`).
"""

import dataclasses
from pathlib import Path
from typing import Any, Optional

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    CommChannel,
    CouplingConfig,
    InProcessCommChannel,
    MpiTransport,
    ParticipantExchanges,
    comm_channel_timeout,
    run_participants,
)
from neofoam.framework.coupling.mpi import RegionComm
from neofoam.framework.solver import Solver, SolverSpec
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.regions import ControlDictConfig

from .mpi import (
    RegionLayout,
    abort,
    abort_all,
    check_decomposed,
    check_size,
    check_world,
    failure,
    import_mpi,
    keep_alive,
    under_mpi,
)

multiRun = Solver("multiRun")

multiRun.config(ControlDictConfig)


def couplings(
    regions: list[str], comm_channel: Optional[CommChannel]
) -> dict[str, ParticipantExchanges]:
    """Each region's coupling from ``system/coupling.yaml``; none without it."""
    if not CouplingConfig.get_default_path().exists():
        return {}
    return CouplingConfig.load().participant_exchanges(regions, comm_channel)


def run() -> dict[str, Context]:
    """Run the multi-region case in the working directory; return each region's final Context.

    Example::

        cd <case>
        python -m neofoam.solver.multiRun
        mpirun -np 3 python -m neofoam.solver.multiRun
    """
    config = ControlDictConfig.load(case_dir=Path("."))
    if under_mpi():
        return run_mpi(config)
    return run_threads(config)


def run_threads(config: ControlDictConfig) -> dict[str, Context]:
    """Run every region of the case on a thread of its own, coupled in-process."""
    regions = list(config.participants)
    comm_channel = InProcessCommChannel(timeout=comm_channel_timeout())
    coupling = couplings(regions, comm_channel)
    runtimes = {
        region: resolve_solver(name).instantiate(region=region, coupling=coupling.get(region))
        for region, name in config.solvers().items()
    }
    tasks = {region: runtime.run for region, runtime in runtimes.items()}
    return run_participants(tasks, comm_channel)


def run_mpi(config: ControlDictConfig) -> dict[str, Context]:
    """Run this process's region of the case, under MPI; return its final Context.

    Every process checks the case first; rank 0 reports what does not fit and
    the run aborts. An error on any process later aborts every process
    (``MPI_Abort``), so none waits for it. MPI is finalized at exit.
    """
    mpi = import_mpi()
    world = mpi.COMM_WORLD
    layout = RegionLayout(config.processes())
    problem = None
    specs: dict[str, SolverSpec] = {}
    planned: Optional[ParticipantExchanges] = None
    timeout: Optional[float] = None
    try:
        check_size(layout, world.size)
        check_decomposed(layout)
        specs = {region: resolve_solver(name) for region, name in config.solvers().items()}
        region = layout.region_of(world.rank)
        planned = couplings(list(config.participants), None).get(region)
        timeout = comm_channel_timeout()
    except Exception as error:
        problem = failure(error)
    problems = [found for found in world.allgather(problem) if found is not None]
    if problems:
        abort_all(world, problems[0])
    region = layout.region_of(world.rank)
    # a channel of its own: no other message meets the coupling's
    channel = world.Dup()
    comm = world.Split(layout.index(region), key=world.rank)
    try:
        coupling = mpi_coupling(planned, layout, channel, comm, timeout)
        runtime = specs[region].instantiate(
            argv=layout.argv(region), region=region, coupling=coupling
        )
        ctx = runtime.initialize()
        keep_alive(mpi, ctx)
        check_world(region, comm)
        runtime.resolve_operations(ctx).run(ctx)
        if coupling is not None and coupling.comm_channel is not None:
            coupling.comm_channel.close(region)
    except BaseException as error:
        abort(world, f"region {region} (rank {comm.rank} of {comm.size}) failed:\n{failure(error)}")
    world.Barrier()
    return {region: ctx}


def mpi_coupling(
    planned: Optional[ParticipantExchanges],
    layout: RegionLayout,
    channel: Any,
    comm: Any,
    timeout: Optional[float],
) -> Optional[ParticipantExchanges]:
    """The coupling of a region on MPI processes, from ``planned``; ``None`` if not coupled.

    Only the region's root gets a comm channel, over ``channel``.
    """
    if planned is None:
        return None
    comm_channel = None
    if comm.rank == 0:
        roots = {other: layout.first(other) for other in planned.group}
        comm_channel = MpiTransport(channel, roots, planned.region, timeout=timeout)
    region_comm = RegionComm(comm) if comm.size > 1 else None
    return dataclasses.replace(planned, comm_channel=comm_channel, comm=region_comm)
