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
from neofoam.framework.coupling.mpi import FollowerTransport, RegionComm
from neofoam.framework.solver import Solver, SolverSpec
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.regions import ControlDictConfig

from .mpi import (
    RegionLayout,
    abort,
    abort_all,
    check_decomposed,
    check_size,
    check_sub_communicator,
    check_world,
    failure,
    import_mpi,
    keep_alive,
    under_mpi,
)

multiRun = Solver("multiRun")

multiRun.config(ControlDictConfig)


def couplings(regions: list[str], comm_channel: CommChannel) -> dict[str, ParticipantExchanges]:
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

    Every process first checks the case against the run: the number of
    processes, the decomposition of each region into its processes (every
    region runs ``-parallel -world <region>``), that its solver can run on part
    of the processes, the coupling and its comm channel timeout. If one fails,
    rank 0 prints why and the run aborts. Then MPI_COMM_WORLD is split into the
    regions (:class:`~neofoam.solver.multiRun.mpi.RegionLayout`), and each
    process runs its region. A coupled region talks to the others through its
    root (:func:`mpi_coupling`). An error on any process prints its traceback
    and aborts every process (``MPI_Abort``), so no process waits for it. The
    region's Context outlives the run; MPI is finalized at exit.
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
        check_sub_communicator(layout, specs)
        region = layout.region_of(world.rank)
        planned = couplings(list(config.participants), FollowerTransport(region)).get(region)
        timeout = comm_channel_timeout()
    except Exception as error:
        problem = failure(error)
    problems = [found for found in world.allgather(problem) if found is not None]
    if problems:
        abort_all(world, problems[0])
    region = layout.region_of(world.rank)
    # the coupling's channel between the regions' roots: no other message meets its own
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
        if coupling is not None:
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

    On the region's root (rank 0 of ``comm``) an
    :class:`~neofoam.framework.coupling.MpiTransport` over ``channel`` to the
    roots of the region's group, with ``timeout``; on its other processes a
    :class:`~neofoam.framework.coupling.mpi.FollowerTransport`, which talks to no
    region. Each has the region's processes as its ``comm``
    (:class:`~neofoam.framework.coupling.mpi.RegionComm`).
    """
    if planned is None:
        return None
    region = planned.region
    comm_channel: CommChannel = FollowerTransport(region)
    if comm.rank == 0:
        roots = {other: layout.first(other) for other in planned.group}
        comm_channel = MpiTransport(channel, roots, region, timeout=timeout)
    return dataclasses.replace(planned, comm_channel=comm_channel, comm=RegionComm(comm))
