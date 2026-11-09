# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling of regions: each region runs its own solver, regions share only a comm channel.

A ``@coupled`` class declares what it sends and receives (``FaceData``,
``CellData``, ``ValueData`` models); its owner binds it to a location
(``coupling(ctx).add(Patch("top"), bc)``), which is its port. An
:class:`Exchange` pairs what one port sends with what the other receives, its
mapping moves the data between their locations. The region's :class:`Participant`
model builds one coupling op per :class:`CouplingLoop`; a loop's leader (its last
region in stepping order) decides convergence and the final iteration.
"""

from .comm.channel import (
    CommChannel,
    CommChannelClosed,
    CommChannelError,
    CouplingDeadlock,
    CouplingTimeout,
)
from .comm.in_process import InProcessCommChannel
from .config import CouplingConfig, comm_channel_timeout
from .coupled import CoupledPort, coupled
from .data import Array, CellData, FaceData, ProbeData, Scalar, ValueData
from .exchanges import CouplingLoop, Exchange, ParticipantExchanges, participant_exchanges
from .field_port import FieldAccess, FieldPort
from .location import Cells, Geometry, Global, MeshGeometry, Patch
from .loop._leader import CouplingNotConvergedWarning
from .model import Participant, coupling
from .mpi import MpiTransport
from .run import run_participants
from .strategies.acceleration import AitkenRelaxation, ConstantRelaxation, CouplingAcceleration
from .strategies.convergence import CouplingConvergence, RelativeConvergence, StreamChange
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
    "FieldAccess",
    "FieldPort",
    "Geometry",
    "Global",
    "IdentityMapping",
    "InProcessCommChannel",
    "MeshGeometry",
    "MpiTransport",
    "NearestNeighborMapping",
    "Patch",
    "ProbeData",
    "ProbeMapping",
    "ParticipantExchanges",
    "CouplingConfig",
    "RelativeConvergence",
    "Scalar",
    "StreamChange",
    "UniformMapping",
    "ValueData",
    "VolumeIntegralMapping",
    "VolumeMeanMapping",
    "comm_channel_timeout",
    "coupled",
    "coupling",
    "participant_exchanges",
    "run_participants",
]
