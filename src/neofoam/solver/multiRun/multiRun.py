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
    CouplingNode,
    GroupSide,
    InProcessCommChannel,
    Link,
    RegionCoupling,
    node_controller,
    region_couplings,
    run_regions,
)
from neofoam.framework.solver import Solver, SolverRuntime
from neofoam.framework.solver.registry import resolve_solver
from neofoam.framework.types import Placement

from .configs import ControlDictConfig, CouplingNodeConfig, RegionCouplingConfig

multiRun = Solver("multiRun")

multiRun.config(ControlDictConfig)


def couplings(regions: list[str], comm_channel: CommChannel) -> dict[str, RegionCoupling]:
    """The coupling of every coupled region, as ``system/regionCoupling.yaml`` lists it.

    One link per entry of ``exchanges``; one coupling node per entry of
    ``couplings``, outer to inner (the ops ``<region>.<node>``), each with its
    convergence, maxIterations, an acceleration per exchange that names one (an
    instance each) and its placement per region. Without ``couplings`` one node
    ``coupling`` holds every exchange (see :meth:`RegionCouplingConfig.nodes`).
    The scheme is the coupling's. Without the file no region is coupled.
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
    for name, exchange in zip(config.exchange_names(), config.exchanges):
        first, second = exchange.sides.items()
        links.append(Link(name, (first, second), exchange.mapping))
    nodes = [_node(node) for node in config.nodes()]
    return region_couplings(
        regions,
        links,
        comm_channel,
        parallel=config.scheme == "parallel",
        controller=_controller(config),
        nodes=nodes,
    )


def _node(config: CouplingNodeConfig) -> CouplingNode:
    """The coupling node ``config`` describes."""
    return CouplingNode(
        config.name,
        tuple(exchange.name for exchange in config.exchanges),
        max_iterations=config.maxIterations,
        convergence=tuple(CONVERGENCES[e.type](e.data, e.limit) for e in config.convergence),
        acceleration={
            exchange.name: ACCELERATIONS[exchange.acceleration.type](
                exchange.acceleration.relaxation
            )
            for exchange in config.exchanges
            if exchange.acceleration is not None
        },
        placement={
            region: Placement(
                None if wraps.operations is None else tuple(wraps.operations), wraps.downstream
            )
            for region, wraps in config.wraps.items()
        },
    )


def _controller(config: RegionCouplingConfig) -> Any:
    """Builds the leader's controller as the node describes, once per node and group.

    Raises ``ValueError`` when a ``convergence`` entry of the node names data no
    side of the node sends (by the field the side couples).
    """
    checked = {node.name: {entry.data for entry in node.convergence} for node in config.nodes()}

    def controller(
        sides: list[GroupSide], label: str, *, parallel: bool, node: CouplingNode
    ) -> CouplingController:
        sent = {side.field for side in sides}
        unknown = sorted(checked.get(node.name, set()) - sent)
        if unknown:
            raise ValueError(
                f"system/regionCoupling.yaml: convergence on {unknown}, which no side of the "
                f"group of {label} sends (its sides solve {sorted(sent)})"
            )
        return node_controller(sides, label, parallel=parallel, node=node)

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
