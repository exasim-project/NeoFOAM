# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling of regions: each region runs its own solver, regions share only a comm channel.

A ``@coupled`` class declares what it sends and receives (``FaceData``,
``CellData``, ``ValueData`` models); its owner binds it to a location
(``coupling(ctx).add(Patch("top"), bc)``), which is its port. An
:class:`Exchange` pairs what one port sends with what the other receives, its
mapping moves the data between their locations. The region's :class:`Participant`
model builds one coupling op per :class:`CouplingLoop`; a loop's leader (its last
region in stepping order) decides convergence and the final pass.
"""

from .comm.channel import (
    CommChannel,
    CommChannelClosed,
    CommChannelError,
    CouplingDeadlock,
    CouplingTimeout,
)
from .comm.in_process import InProcessCommChannel
from .coupled import CoupledPort, coupled
from .data import Array, CellData, FaceData, ProbeData, Scalar, ValueData
from .exchanges import CouplingLoop, Exchange, ParticipantExchanges, participant_exchanges
from .iteration._leader import CouplingNotConvergedWarning
from .location import Cells, Geometry, Global, MeshGeometry, Patch
from .model import Participant, coupling
from .run import run_participants
from .strategies.acceleration import AitkenRelaxation, ConstantRelaxation, CouplingAcceleration
from .strategies.convergence import CouplingConvergence, RelativeConvergence, SideChange
from .strategies.mapping import (
    AreaIntegralMapping,
    AreaMeanMapping,
    BoundMapping,
    CouplingMapping,
    IdentityMapping,
    NearestNeighborMapping,
    ProbeMapping,
    UniformMapping,
    VolumeIntegralMapping,
    VolumeMeanMapping,
)

__all__ = [
    "AitkenRelaxation",
    "AreaIntegralMapping",
    "AreaMeanMapping",
    "Array",
    "BoundMapping",
    "CellData",
    "Cells",
    "CommChannel",
    "CommChannelClosed",
    "CommChannelError",
    "ConstantRelaxation",
    "CoupledPort",
    "Participant",
    "CouplingAcceleration",
    "CouplingConvergence",
    "CouplingDeadlock",
    "CouplingMapping",
    "CouplingLoop",
    "CouplingNotConvergedWarning",
    "CouplingTimeout",
    "Exchange",
    "FaceData",
    "Geometry",
    "Global",
    "IdentityMapping",
    "InProcessCommChannel",
    "MeshGeometry",
    "NearestNeighborMapping",
    "Patch",
    "ProbeData",
    "ProbeMapping",
    "ParticipantExchanges",
    "RelativeConvergence",
    "Scalar",
    "SideChange",
    "UniformMapping",
    "ValueData",
    "VolumeIntegralMapping",
    "VolumeMeanMapping",
    "coupled",
    "coupling",
    "participant_exchanges",
    "run_participants",
]
