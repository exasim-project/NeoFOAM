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

import math
import os
from functools import partial
from pathlib import Path
from typing import Optional

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    CommChannel,
    CouplingNode,
    InProcessCommChannel,
    Link,
    RegionCoupling,
    load_plugins,
)
from neofoam.framework.coupling.comm_channel import DEFAULT_TIMEOUT
from neofoam.framework.coupling.region import region_couplings, run_regions
from neofoam.framework.solver import Solver, SolverRuntime
from neofoam.framework.solver.registry import resolve_solver
from neofoam.framework.types import Placement

from .configs import ControlDictConfig, CouplingNodeConfig, RegionCouplingConfig

multiRun = Solver("multiRun")

multiRun.config(ControlDictConfig)


#: The environment variable that sets the comm channel timeout, in seconds, over the case's.
TIMEOUT_VARIABLE = "NEOFOAM_COUPLING_TIMEOUT"


def transport_timeout() -> Optional[float]:
    """How long a region waits for a message, in seconds; ``None``: without end.

    ``NEOFOAM_COUPLING_TIMEOUT`` if it is set, else ``transportTimeout`` of
    ``system/regionCoupling.yaml``, else 600 seconds; ``0`` waits without end.
    Raises ``ValueError`` when the variable holds no number of seconds of at least 0.
    """
    value = os.environ.get(TIMEOUT_VARIABLE)
    if value is not None:
        try:
            seconds = float(value)
            if not (math.isfinite(seconds) and seconds >= 0):
                raise ValueError(value)
        except ValueError:
            raise ValueError(
                f"{TIMEOUT_VARIABLE}={value!r}: the comm channel timeout is a number of "
                "seconds, at least 0 (0: no timeout)"
            ) from None
    elif RegionCouplingConfig.get_default_path().exists():
        configured = RegionCouplingConfig.load().transportTimeout
        seconds = 0.0 if configured is None else configured
    else:
        seconds = DEFAULT_TIMEOUT
    return seconds or None


def couplings(regions: list[str], comm_channel: CommChannel) -> dict[str, RegionCoupling]:
    """The coupling of every coupled region, as ``system/regionCoupling.yaml`` lists it.

    One link per entry of ``exchanges``; one coupling node per entry of
    ``couplings``, outer to inner (the ops ``<region>.<node>``), each with its
    convergence, maxIterations, an acceleration per exchange that names one (an
    instance each), its scheme and its placement per region. Without
    ``couplings`` one node ``coupling`` holds every exchange (see
    :meth:`RegionCouplingConfig.nodes`). Without the file no region is coupled.
    Raises ``ValueError`` when a serial node encloses a node coupling two of its
    regions as well (:func:`~neofoam.framework.coupling.region.region_couplings`).
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
        nodes=nodes,
    )


def _node(config: CouplingNodeConfig) -> CouplingNode:
    """The coupling node ``config`` describes."""
    return CouplingNode(
        config.name,
        tuple(exchange.name for exchange in config.exchanges),
        max_iterations=config.maxIterations,
        scheme=config.scheme or "parallel",
        convergence=tuple(config.convergence),
        acceleration={
            exchange.name: exchange.acceleration.model_copy(deep=True)
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


def run_region(runtime: SolverRuntime) -> Context:
    """Run one region as its solver runs alone: initialize, resolve, run, finalize."""
    ctx = runtime.initialize()
    runtime.resolve_operations(ctx).run(ctx)
    runtime.finalize(ctx)
    return ctx


def run() -> dict[str, Context]:
    """Run the multi-region case in the working directory; return each region's final Context.

    The coupling plugins are loaded first, before any region starts
    (:func:`~neofoam.framework.coupling.load_plugins`), so the case's
    ``system/regionCoupling.yaml`` may name their strategies.

    Example::

        cd <case>
        python -m neofoam.solver.multiRun
    """
    load_plugins()
    config = ControlDictConfig.load(case_dir=Path("."))
    regions = list(config.regionSolvers)
    comm_channel = InProcessCommChannel(timeout=transport_timeout())
    coupling = couplings(regions, comm_channel)
    runtimes = {
        region: resolve_solver(name).instantiate(region=region, coupling=coupling.get(region))
        for region, name in config.regionSolvers.items()
    }
    tasks = {region: partial(run_region, runtime) for region, runtime in runtimes.items()}
    return run_regions(tasks, comm_channel)
