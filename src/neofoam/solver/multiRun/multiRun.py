# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun: one solver per region in one case, each on its own thread.

``system/controlDict.yaml`` lists the region solvers; each region is an ordinary
solver instantiated with ``region=``. Regions share nothing but a
:class:`~neofoam.framework.coupling.CommChannel`, over which the coupled patches
listed in ``system/regionCoupling.yaml`` exchange their data.
"""

from pathlib import Path
from typing import Optional

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    CommChannel,
    InProcessCommChannel,
    ParticipantExchanges,
    RegionCouplingConfig,
)
from neofoam.framework.coupling.comm.channel import DEFAULT_TIMEOUT
from neofoam.framework.coupling.run import run_participants
from neofoam.framework.solver import Solver
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.regions import ControlDictConfig

multiRun = Solver("multiRun")

multiRun.config(ControlDictConfig)


def comm_channel_timeout() -> Optional[float]:
    """Seconds a region waits for a message (``commChannelTimeout``); ``None``: no end."""
    if not RegionCouplingConfig.get_default_path().exists():
        return DEFAULT_TIMEOUT
    return RegionCouplingConfig.load().commChannelTimeout or None


def couplings(regions: list[str], comm_channel: CommChannel) -> dict[str, ParticipantExchanges]:
    """Each region's coupling from ``system/regionCoupling.yaml``; none without it."""
    if not RegionCouplingConfig.get_default_path().exists():
        return {}
    return RegionCouplingConfig.load().participant_exchanges(regions, comm_channel)


def run() -> dict[str, Context]:
    """Run the multi-region case in the working directory; return each region's final Context.

    Example::

        cd <case>
        python -m neofoam.solver.multiRun
    """
    config = ControlDictConfig.load(case_dir=Path("."))
    regions = list(config.regionSolvers)
    comm_channel = InProcessCommChannel(timeout=comm_channel_timeout())
    coupling = couplings(regions, comm_channel)
    runtimes = {
        region: resolve_solver(name).instantiate(region=region, coupling=coupling.get(region))
        for region, name in config.regionSolvers.items()
    }
    return run_participants(
        {region: runtime.run for region, runtime in runtimes.items()}, comm_channel
    )
