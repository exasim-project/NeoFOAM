# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling of regions: each region runs its own solver, regions share only a comm channel.

A ``@coupled`` boundary condition declares its side; the region's :class:`Coupling`
model builds one coupling op per :class:`CouplingNode`, a loop whose test exchanges
the sides' data. A node's leader (its last region in stepping order) decides
convergence and the final pass.
"""

from .comm_channel import (
    CommChannel,
    CommChannelClosed,
    CommChannelError,
    CouplingDeadlock,
    CouplingTimeout,
    InProcessCommChannel,
)
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
from .region import CouplingNode, Exchange, RegionCoupling

__all__ = [
    "AitkenRelaxation",
    "Array",
    "BoundMapping",
    "CommChannel",
    "CommChannelClosed",
    "CommChannelError",
    "ConstantRelaxation",
    "Coupling",
    "CouplingAcceleration",
    "CouplingConvergence",
    "CouplingDeadlock",
    "CouplingMapping",
    "CouplingNode",
    "CouplingNotConvergedWarning",
    "CouplingTimeout",
    "Exchange",
    "IdentityMapping",
    "InProcessCommChannel",
    "NearestNeighborMapping",
    "RegionCoupling",
    "RelativeConvergence",
    "SideChange",
    "coupled",
]
