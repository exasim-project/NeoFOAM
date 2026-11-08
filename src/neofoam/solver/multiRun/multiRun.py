# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun — several solvers, one per region, in one case: a launcher.

The multi-region layout of OpenFOAM.org's ``foamMultiRun``: ``system/controlDict.yaml``
lists the region solvers (:class:`~neofoam.solver.multiRun.configs.ControlDictConfig`)
and each region reads its own files from its region directories. Every region is
an ordinary solver instantiated with ``region=``: it builds its own graph and
runs its own time loop, on its own thread. Regions share nothing but a
:class:`~neofoam.framework.coupling.CommChannel`: the coupled patches of the
regions exchange their data through it as ``system/regionCoupling.yaml`` lists
(:func:`couplings`).
"""

from functools import partial
from pathlib import Path
from typing import Any

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    ACCELERATIONS,
    CONVERGENCES,
    CommChannel,
    CouplingController,
    GroupSide,
    InProcessCommChannel,
    Link,
    RegionCoupling,
    region_couplings,
    run_regions,
)
from neofoam.framework.solver import Solver, SolverRuntime
from neofoam.framework.solver.registry import resolve_solver

from .configs import ControlDictConfig, RegionCouplingConfig

multiRun = Solver("multiRun")

multiRun.config(ControlDictConfig)


def couplings(regions: list[str], comm_channel: CommChannel) -> dict[str, RegionCoupling]:
    """The coupling of every coupled region, as ``system/regionCoupling.yaml`` lists it.

    One link per entry of ``exchanges``; the file's scheme, acceleration,
    convergence and maxIterations go to every group's controller. Without the
    file no region is coupled.
    """
    if not RegionCouplingConfig.get_default_path().exists():
        return {}
    config = RegionCouplingConfig.load()
    named = sorted({region for exchange in config.exchanges for region in exchange.sides})
    unknown = [region for region in named if region not in regions]
    if unknown:
        raise ValueError(
            f"system/regionCoupling.yaml couples the regions {unknown}, "
            f"which are not in regionSolvers {regions}"
        )
    links: list[Link] = []
    for exchange in config.exchanges:
        ends = tuple(exchange.sides.items())
        name = exchange.name or _free("-".join(region for region, _patch in ends), links)
        if any(link.name == name for link in links):
            raise ValueError(f"system/regionCoupling.yaml: two exchanges are named '{name}'")
        links.append(Link(name, (ends[0], ends[1]), exchange.mapping))
    return region_couplings(
        regions,
        links,
        comm_channel,
        parallel=config.scheme == "parallel",
        controller=_controller(config),
    )


def _free(name: str, links: list[Link]) -> str:
    """``name``, numbered if a link has it."""
    taken = {link.name for link in links}
    free, number = name, 1
    while free in taken:
        number += 1
        free = f"{name}-{number}"
    return free


def _controller(config: RegionCouplingConfig) -> Any:
    """Builds the leader's controller as the config describes, once per group.

    Raises ``ValueError`` when a ``convergence`` entry names data no side of the
    group sends (by the field the side couples).
    """

    def controller(sides: list[GroupSide], label: str, *, parallel: bool) -> CouplingController:
        sent = {side.field for side in sides}
        unknown = sorted({entry.data for entry in config.convergence} - sent)
        if unknown:
            raise ValueError(
                f"system/regionCoupling.yaml: convergence on {unknown}, which no side of the "
                f"group of {label} sends (its sides solve {sorted(sent)})"
            )
        measures = [CONVERGENCES[e.type](e.data, e.limit) for e in config.convergence]
        return CouplingController(
            sides,
            label,
            converged=lambda changes: all(measure(changes) for measure in measures),
            max_iterations=config.maxIterations,
            parallel=parallel,
            acceleration=ACCELERATIONS[config.acceleration.type](config.acceleration.relaxation),
        )

    return controller


def run_region(runtime: SolverRuntime) -> Context:
    """Run one region as its solver runs alone: initialize, resolve, run, finalize."""
    ctx = runtime.initialize()
    runtime.resolve_operations(ctx).run(ctx)
    runtime.finalize(ctx)
    return ctx


def run() -> dict[str, Context]:
    """Run the multi-region case in the working directory; return each region's final Context.

    Example::

        cd <case>
        python -m neofoam.solver.multiRun
    """
    config = ControlDictConfig.load(case_dir=Path("."))
    regions = list(config.regionSolvers)
    comm_channel = InProcessCommChannel()
    coupling = couplings(regions, comm_channel)
    runtimes = {
        region: resolve_solver(name).instantiate(region=region, coupling=coupling.get(region))
        for region, name in config.regionSolvers.items()
    }
    tasks = {region: partial(run_region, runtime) for region, runtime in runtimes.items()}
    return run_regions(tasks, comm_channel)
