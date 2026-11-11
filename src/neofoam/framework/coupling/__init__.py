# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling of regions: the public API of sides, exchanges, nodes, strategies and transports.

- :func:`coupled` — a boundary condition declares its side: the field it couples,
  what it sends and receives (one pydantic model each, of :data:`Array` and
  plain fields) and its points; an instance adds itself to its region's
  :class:`Coupling` model (``Coupling.of(ctx).add(self)``).
- :class:`Coupling` — the coupling of one region, ``ctx.models["coupling"]``:
  its sides, the operations of the region's graph (one coupling op per node,
  :class:`~neofoam.framework.coupling.loop.CouplingOp`) and, in a transient
  case, its time-step op (the group steps the smallest time step any region
  asks for).
- :class:`CouplingNode` — a named coupling loop: its exchanges, its scheme
  (``parallel``, the default, or ``serial``), max_iterations, convergence
  measures, an acceleration per exchange and its placement per region; nodes
  are listed outer to inner. Its leader (the
  :class:`~neofoam.framework.coupling.controller.CouplingController` of the
  node's last region) decides when the node's data converged and relaxes it;
  :class:`CouplingNotConvergedWarning` when ``max_iterations`` ends a time step.
- :class:`RegionCoupling` — what a region's solver knows of the case's
  coupling: its group, the exchanges (:class:`Link`), its nodes and its
  transport (built by :func:`~neofoam.framework.coupling.region.region_couplings`).
- :class:`Transport` — all data between regions goes through it, point to point
  (send, receive, close); :class:`InProcessTransport` between threads, which
  raises :class:`CouplingDeadlock` when every running region waits for a message
  no region sent, and :class:`CouplingTimeout` after its ``timeout`` (both
  :class:`CouplingTransportError`, as :class:`TransportClosed`).
- strategies, each a :class:`~neofoam.core.plugin_system.PluginSystem` family
  selected by ``type``: :class:`CouplingMapping`
  (:class:`NearestNeighborMapping`, the default, :class:`IdentityMapping`; bound once to the
  points of both sides: :data:`BoundMapping`), :class:`CouplingAcceleration`
  (:class:`ConstantRelaxation`, :class:`AitkenRelaxation`) and
  :class:`CouplingConvergence` (:class:`RelativeConvergence`, from the
  :class:`SideChange` of every side).

Everything else (the coupling op, the controller, the sides of a group, the
tags of the messages, ...) is internal: import it from its submodule.
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
