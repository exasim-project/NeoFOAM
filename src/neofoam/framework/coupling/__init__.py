# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling of regions: each region runs its own solver, regions share only a transport.

A ``@coupled`` boundary condition declares its side; the region's
:class:`Coupling` model builds one coupling op per :class:`CouplingNode`, a loop
in the region's own graph whose test exchanges the sides' data with the partner
regions. The leader of a node (its last region in stepping order) decides
convergence and the final pass. Internals come from their submodules.
"""

from .controller import CouplingNotConvergedWarning
from .coupled import coupled
from .data import Array
from .kernels import (
    AitkenRelaxation,
    BoundMapping,
    ConstantRelaxation,
    CouplingAcceleration,
    CouplingConvergence,
    CouplingMapping,
    IdentityMapping,
    NearestNeighborMapping,
    RelativeConvergence,
    SideChange,
)
from .model import Coupling
from .region import CouplingNode, Link, RegionCoupling
from .transport import (
    CouplingDeadlock,
    CouplingTimeout,
    CouplingTransportError,
    InProcessTransport,
    Transport,
    TransportClosed,
)

__all__ = [
    "AitkenRelaxation",
    "Array",
    "BoundMapping",
    "ConstantRelaxation",
    "Coupling",
    "CouplingAcceleration",
    "CouplingConvergence",
    "CouplingDeadlock",
    "CouplingMapping",
    "CouplingNode",
    "CouplingNotConvergedWarning",
    "CouplingTimeout",
    "CouplingTransportError",
    "IdentityMapping",
    "InProcessTransport",
    "Link",
    "NearestNeighborMapping",
    "RegionCoupling",
    "RelativeConvergence",
    "SideChange",
    "Transport",
    "TransportClosed",
    "coupled",
]
