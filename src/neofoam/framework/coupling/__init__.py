# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupling of regions: sides, exchanges, mappings, the coupling ops and their transport.

- :func:`coupled` — a boundary condition declares its side: the field it couples,
  what it sends and receives (one pydantic model each, of :data:`Array` and
  plain fields) and its points; an instance adds itself to its region's
  :class:`Coupling` model (``Coupling.of(ctx).add(self)``).
- :class:`Coupling` — the coupling of one region, ``ctx.models["coupling"]``:
  its sides, the operations of the region's graph (one coupling op per node)
  and, in a transient case, its time-step op (the group steps the smallest time
  step any region asks for).
- :class:`CouplingNode` — a named coupling loop: its exchanges, its scheme
  (``parallel``, the default, or ``serial``), max_iterations, convergence
  measures, an acceleration per exchange and its placement per region; nodes
  are listed outer to inner.
- :class:`RegionCoupling` — what a region knows of the case's coupling: its
  group, the exchanges (:class:`Link`), its nodes and its transport;
  :func:`region_couplings` builds it for every coupled region; :func:`run_regions`
  runs each region on its own thread.
- :class:`CouplingOp` — a region's part of a node, ``<region>.<node>`` (its
  :class:`Coupling` builds it): a loop placed as the node says; its loop test
  exchanges the node's sides' data with the partner regions, mapped onto the
  sides' points (:class:`Exchange`, :class:`Mapping`).
- :class:`CouplingController` — the leader's part, in the coupling op of the
  node's last region: decides when the node's data converged, which pass is
  the final one (one more after converging, or the last of max_iterations), and
  relaxes the data, per exchange (built by :func:`node_controller` unless the case gives its own
  factory); :class:`CouplingOutcome` per time step,
  :class:`CouplingNotConvergedWarning` when ``max_iterations`` ends it.
- :class:`Transport` — all data between regions goes through it, point to point
  (send, receive, close); :class:`InProcessTransport` between threads.
- numpy kernels: :func:`nearest_neighbor`, :class:`ConstantRelaxation`,
  :class:`AitkenRelaxation`, :class:`RelativeConvergence`; the registries
  :data:`MAPPINGS`, :data:`ACCELERATIONS` and :data:`CONVERGENCES` name them for
  the case; a package adds its own through the entry-point group
  ``neofoam.coupling`` (:func:`load_plugins`).
"""

from .coupled import coupled
from .kernels import (
    ACCELERATIONS,
    CONVERGENCES,
    ENTRY_POINT_GROUP,
    MAPPINGS,
    AitkenRelaxation,
    ConstantRelaxation,
    Mapping,
    NearestNeighborMapping,
    Registry,
    RelativeConvergence,
    SideChange,
    load_plugins,
    mapping,
    nearest_neighbor,
)
from .loop import (
    Array,
    CoupledSide,
    CouplingController,
    CouplingNotConvergedWarning,
    CouplingOp,
    CouplingOutcome,
    Exchange,
    ExchangeSide,
    GroupSide,
    relative_changes,
    unchanged,
)
from .model import Coupling
from .region import (
    ControllerFactory,
    CouplingNode,
    Link,
    RegionCoupling,
    node_controller,
    region_couplings,
    run_regions,
)
from .transport import InProcessTransport, SideKey, Transport, TransportClosed

__all__ = [
    "ACCELERATIONS",
    "CONVERGENCES",
    "ENTRY_POINT_GROUP",
    "MAPPINGS",
    "AitkenRelaxation",
    "Array",
    "ConstantRelaxation",
    "ControllerFactory",
    "CoupledSide",
    "Coupling",
    "CouplingController",
    "CouplingNode",
    "CouplingNotConvergedWarning",
    "CouplingOp",
    "CouplingOutcome",
    "Exchange",
    "ExchangeSide",
    "GroupSide",
    "InProcessTransport",
    "Link",
    "Mapping",
    "NearestNeighborMapping",
    "RegionCoupling",
    "Registry",
    "RelativeConvergence",
    "SideChange",
    "SideKey",
    "Transport",
    "TransportClosed",
    "coupled",
    "load_plugins",
    "mapping",
    "nearest_neighbor",
    "node_controller",
    "region_couplings",
    "relative_changes",
    "run_regions",
    "unchanged",
]
