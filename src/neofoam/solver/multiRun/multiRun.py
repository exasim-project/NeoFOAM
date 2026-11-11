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
from neofoam.framework.coupling import CommChannel, InProcessCommChannel, RegionCoupling
from neofoam.framework.coupling.comm_channel import DEFAULT_TIMEOUT
from neofoam.framework.coupling.region import region_couplings, run_regions
from neofoam.framework.solver import Solver
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.regions import ControlDictConfig

from .configs import RegionCouplingConfig

multiRun = Solver("multiRun")

multiRun.config(ControlDictConfig)


def transport_timeout() -> Optional[float]:
    """How long a region waits for a message, in seconds; ``None``: without end.

    The case's ``transportTimeout``, else 600 seconds; ``0`` waits without end.
    """
    if not RegionCouplingConfig.get_default_path().exists():
        return DEFAULT_TIMEOUT
    return RegionCouplingConfig.load().transportTimeout or None


def couplings(regions: list[str], comm_channel: CommChannel) -> dict[str, RegionCoupling]:
    """The coupling of every coupled region as ``system/regionCoupling.yaml`` lists it.

    Without the file no region is coupled. Raises ``ValueError`` as
    :func:`~neofoam.framework.coupling.region.region_couplings`.
    """
    if not RegionCouplingConfig.get_default_path().exists():
        return {}
    config = RegionCouplingConfig.load()
    return region_couplings(regions, config.links(), comm_channel, config.nodes())


def run() -> dict[str, Context]:
    """Run the multi-region case in the working directory; return each region's final Context.

    Example::

        cd <case>
        python -m neofoam.solver.multiRun
    """
    config = ControlDictConfig.load(case_dir=Path("."))
    regions = list(config.regionSolvers)
    comm_channel = InProcessCommChannel(timeout=transport_timeout())
    coupling = couplings(regions, comm_channel)
    runtimes = {
        region: resolve_solver(name).instantiate(region=region, coupling=coupling.get(region))
        for region, name in config.regionSolvers.items()
    }
    return run_regions({region: runtime.run for region, runtime in runtimes.items()}, comm_channel)
