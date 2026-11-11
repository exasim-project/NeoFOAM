# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What one region knows of the coupling of a case: :class:`RegionCoupling`.

A coupled case lists its exchanges (:class:`Link`, from
``system/regionCoupling.yaml``), its coupling nodes (:class:`CouplingNode`) and
its regions in stepping order (``regionSolvers``). A node is a named coupling
loop: the exchanges it repeats, how it decides they converged, how it relaxes
them and where it sits in each region's graph; every exchange is in one node,
the nodes are listed outer to inner. :func:`region_couplings` turns them into
one :class:`RegionCoupling` per coupled region, plain data plus the region's
transport, handed to the region's runtime (``instantiate(region=...,
coupling=...)``); a solver that supports coupling keeps it in the region's
coupling model (``ctx.models["coupling"]``, a
:class:`~neofoam.framework.coupling.Coupling`), which builds one coupling op
per node of the region.

Regions that exchanges connect, directly or over other regions, form a group;
they agree on the time step (``Coupling.time_step``) and its last region in
stepping order leads that. The regions a node's exchanges connect form the node's group; its
last region in stepping order leads the node (decides when it converged).

Each node has its scheme (``CouplingNode.scheme``): ``parallel`` (the default)
or ``serial``. A serial pass of a region waits for the whole pass of the regions
that step before it, so a serial node may neither enclose a node coupling two of
its regions as well (:func:`region_couplings` raises) nor hold the time step of
its regions (``Coupling.check`` raises): both need the partner to come back
inside the pass.

Example::

    links = [Link("interface", (("left", "right"), ("right", "left")))]
    nodes = [CouplingNode("thermal", ("interface",), max_iterations=20)]
    couplings = region_couplings(["left", "right"], links, InProcessTransport(), nodes=nodes)
    couplings["right"].is_leader    # True
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping, Optional, Protocol, Sequence, TypeVar

from neofoam.framework.types import Placement

from . import kernels
from .controller import CouplingController, GroupSide
from .data import End
from .kernels import CouplingAcceleration, CouplingConvergence
from .transport import Transport, TransportClosed

T = TypeVar("T")


@dataclass(frozen=True)
class Link:
    """One exchange of a case: its name, its two ends ``(region, patch)`` and its mapping.

    ``mapping`` is a :class:`~neofoam.framework.coupling.CouplingMapping`, by default
    :func:`~neofoam.framework.coupling.kernels.default_mapping` (nearest-neighbor: both
    patches need coordinates; :class:`~neofoam.framework.coupling.IdentityMapping` for
    patches that share their points).
    """

    name: str
    ends: tuple[End, End]
    mapping: kernels.CouplingMapping = field(default_factory=kernels.default_mapping)

    def partner(self, region: str) -> End:
        """The end of the exchange that is not in ``region``."""
        first, second = self.ends
        return second if first[0] == region else first


@dataclass(frozen=True)
class CouplingNode:
    """A named coupling loop: its exchanges, its convergence and acceleration, its placement.

    Each region the node's ``exchanges`` (names of :class:`Link` s) touch gets
    one coupling op ``<region>.<name>`` for it. Once every measure of
    ``convergence`` holds (none: :func:`~neofoam.framework.coupling.controller.unchanged`)
    the node's leader runs one more pass, the final one, and stops after it; the
    last of ``max_iterations`` passes is final regardless (an operation marked
    ``final`` runs in the final pass only). ``acceleration`` relaxes the data
    carried into a pass, the final one too, at most one per exchange (by its
    name; an exchange without one is not relaxed).
    ``placement`` places the node's op in a region's graph (by region;
    :class:`~neofoam.framework.types.Placement` ``()`` for a region it does not
    name: around the whole body of the top scope, or of the node before).
    ``scheme`` ``parallel``: a side receives what its partner sent in the pass
    before; ``serial``: from a region that steps earlier, what it sent in the
    same pass.

    An acceleration keeps the data of its last passes: give each node its own.
    """

    name: str
    exchanges: tuple[str, ...]
    max_iterations: int = 100
    convergence: tuple[CouplingConvergence, ...] = ()
    acceleration: Mapping[str, CouplingAcceleration] = field(default_factory=dict)
    placement: Mapping[str, Placement] = field(default_factory=dict)
    scheme: Literal["serial", "parallel"] = "parallel"

    @property
    def parallel(self) -> bool:
        """Whether the node's scheme is ``parallel``."""
        return self.scheme == "parallel"

    def placement_of(self, region: str) -> Placement:
        """Where the node's op sits in the graph of ``region``."""
        return self.placement.get(region, Placement())


class ControllerFactory(Protocol):
    """Builds a node leader's controller from the node's sides, the leader's label and the node.

    ``node`` holds the node's scheme (the controller decides with it which sides
    receive fresh or carried data), convergence, acceleration and max_iterations.
    """

    def __call__(
        self, sides: Sequence[GroupSide], label: str, /, *, node: CouplingNode
    ) -> CouplingController: ...


def node_controller(
    sides: Sequence[GroupSide], label: str, /, *, node: CouplingNode
) -> CouplingController:
    """The default :class:`ControllerFactory`: a controller as ``node`` describes it."""
    return CouplingController(
        sides,
        label,
        convergence=node.convergence,
        max_iterations=node.max_iterations,
        parallel=node.parallel,
        acceleration=dict(node.acceleration),
    )


@dataclass(frozen=True)
class RegionCoupling:
    """What region ``region`` knows of the coupling of its case.

    ``group`` holds the regions coupled with it, itself included, in stepping
    order; ``links`` the exchanges of the group; ``nodes`` the nodes with an
    exchange of this region, outer to inner. The leader of each node builds its
    controller with ``controller``, handing it the node.
    """

    region: str
    group: tuple[str, ...]
    links: tuple[Link, ...]
    transport: Transport
    controller: ControllerFactory = node_controller
    nodes: tuple[CouplingNode, ...] = ()

    @property
    def leader(self) -> str:
        """The region of the group that agrees the time step: the last in stepping order."""
        return self.group[-1]

    @property
    def is_leader(self) -> bool:
        """Whether this region leads its group (in ``Coupling.time_step``)."""
        return self.region == self.leader

    @property
    def patches(self) -> tuple[str, ...]:
        """The patches of this region that an exchange connects."""
        return tuple(
            dict(link.ends)[self.region] for link in self.links if self.region in dict(link.ends)
        )

    def node(self, name: str) -> CouplingNode:
        """The node ``name`` of this region."""
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"region {self.region} is in no coupling node '{name}'")

    def node_links(self, node: CouplingNode) -> tuple[Link, ...]:
        """The exchanges of ``node`` in this region's group."""
        return tuple(link for link in self.links if link.name in node.exchanges)

    def node_of(self, patch: str) -> CouplingNode:
        """The node whose exchange connects ``patch`` of this region."""
        for node in self.nodes:
            for link in self.node_links(node):
                if dict(link.ends).get(self.region) == patch:
                    return node
        raise KeyError(f"no coupling node has an exchange on {self.region}.{patch}")

    def node_group(self, node: CouplingNode) -> tuple[str, ...]:
        """The regions ``node`` connects with this region, itself included, in stepping order."""
        return _group(self.region, self.group, self.node_links(node))

    def node_leader(self, node: CouplingNode) -> str:
        """The region that decides for ``node``: the last of its regions in stepping order."""
        return self.node_group(node)[-1]


def region_couplings(
    regions: Sequence[str],
    links: Sequence[Link],
    transport: Transport,
    controller: ControllerFactory = node_controller,
    nodes: Optional[Sequence[CouplingNode]] = None,
) -> dict[str, RegionCoupling]:
    """The coupling of every region ``links`` connect, by region; ``regions`` in stepping order.

    ``nodes`` lists the coupling nodes, outer to inner; without them one node
    ``coupling`` holds every exchange, with the defaults of :class:`CouplingNode`.

    Raises ``ValueError`` when a link names a region not in ``regions``, joins
    two patches of one region, or shares a patch with another link; when two
    nodes have one name, an exchange is in no node or in two, a node names an
    exchange that does not exist, or a node's placement or acceleration names a
    region or exchange that is not the node's; when a serial node encloses a
    node (one listed after it) that couples two of its regions as well: a serial
    pass of a region waits for its partner's whole pass, which needs the inner
    node of both regions.
    """
    named = sorted({region for link in links for region, _patch in link.ends} - set(regions))
    if named:
        raise ValueError(
            f"the exchanges couple the regions {named}, which are not in {list(regions)}"
        )
    owner: dict[End, str] = {}
    names = [link.name for link in links]
    for link in links:
        if names.count(link.name) > 1:
            raise ValueError(f"two exchanges are named '{link.name}'; an exchange's name is unique")
        first, second = link.ends
        if first[0] == second[0]:
            raise ValueError(f"exchange '{link.name}' joins two patches of region {first[0]}")
        for region, patch in link.ends:
            if (region, patch) in owner:
                raise ValueError(
                    f"{region}.{patch} is in the exchanges '{owner[(region, patch)]}' and "
                    f"'{link.name}'; a coupled patch is in one exchange"
                )
            owner[(region, patch)] = link.name
    if nodes is None:
        nodes = [CouplingNode("coupling", tuple(link.name for link in links))]
    _check_nodes(nodes, links)
    couplings: dict[str, RegionCoupling] = {}
    for region in regions:
        if region in couplings:
            continue
        group = _group(region, regions, links)
        if len(group) < 2:
            continue
        group_links = tuple(link for link in links if link.ends[0][0] in group)
        for member in group:
            touching = {link.name for link in group_links if member in dict(link.ends)}
            member_nodes = tuple(node for node in nodes if touching & set(node.exchanges))
            couplings[member] = RegionCoupling(
                member, group, group_links, transport, controller, member_nodes
            )
    return couplings


def _check_nodes(nodes: Sequence[CouplingNode], links: Sequence[Link]) -> None:
    """Raise ``ValueError`` unless ``nodes`` hold every exchange of ``links`` once, consistently."""
    names = [link.name for link in links]
    seen: list[str] = []
    in_node: dict[str, str] = {}
    for node in nodes:
        if node.name in seen:
            raise ValueError(f"two coupling nodes are named '{node.name}'; a node's name is unique")
        seen.append(node.name)
        if not node.exchanges:
            raise ValueError(f"coupling node '{node.name}' has no exchange")
        for exchange in node.exchanges:
            if exchange not in names:
                raise ValueError(
                    f"coupling node '{node.name}' names the exchange '{exchange}', which does "
                    f"not exist; the exchanges are {names}"
                )
            if exchange in in_node:
                raise ValueError(
                    f"exchange '{exchange}' is in the coupling nodes '{in_node[exchange]}' and "
                    f"'{node.name}'; an exchange is in exactly one node"
                )
            in_node[exchange] = node.name
        regions = sorted(
            {region for link in links if link.name in node.exchanges for region, _ in link.ends}
        )
        unknown = sorted(set(node.placement) - set(regions))
        if unknown:
            raise ValueError(
                f"coupling node '{node.name}' places its op in the regions {unknown}, which it "
                f"does not couple; its regions are {regions}"
            )
        foreign = sorted(set(node.acceleration) - set(node.exchanges))
        if foreign:
            raise ValueError(
                f"coupling node '{node.name}' accelerates the exchanges {foreign}, which are "
                f"not its own; its exchanges are {list(node.exchanges)}"
            )
    missing = [name for name in names if name not in in_node]
    if missing:
        raise ValueError(
            f"the exchanges {missing} are in no coupling node; each exchange is in exactly one "
            f"node (the nodes: {seen})"
        )
    _check_serial_nesting(nodes, links)


def _check_serial_nesting(nodes: Sequence[CouplingNode], links: Sequence[Link]) -> None:
    """Raise ``ValueError`` when a serial node encloses a node sharing two of its regions."""

    def regions(node: CouplingNode) -> set[str]:
        return {region for link in links if link.name in node.exchanges for region, _ in link.ends}

    for index, outer in enumerate(nodes):
        if outer.parallel:
            continue
        for inner in nodes[index + 1 :]:
            shared = sorted(regions(outer) & regions(inner))
            if len(shared) >= 2:
                raise ValueError(
                    f"coupling '{outer.name}' is serial but encloses '{inner.name}', which "
                    f"couples {shared} as well; a serial outer pass waits for its partner's "
                    "whole pass, which needs the inner coupling of both - make "
                    f"'{outer.name}' parallel"
                )


def _group(region: str, regions: Sequence[str], links: Sequence[Link]) -> tuple[str, ...]:
    """The regions connected with ``region`` over ``links``, itself included, in stepping order."""
    group, todo = {region}, [region]
    while todo:
        current = todo.pop()
        for link in links:
            ends = {name for name, _patch in link.ends}
            if current in ends:
                todo += sorted(ends - group)
                group |= ends
    return tuple(name for name in regions if name in group)


def run_regions(tasks: Mapping[str, Callable[[], T]], transport: Transport) -> dict[str, T]:
    """Run every region's task on its own thread; return each task's result, by region.

    The threads report to the execution observer of the calling thread. The
    transport learns the running regions (``Transport.start``), so it can tell a
    deadlock. A region whose task fails closes the transport for every region
    (``Transport.abort``), so none waits for it forever; a region that ends
    closes it for itself. Once every thread ended the first error is raised that
    is not a :class:`~neofoam.framework.coupling.TransportClosed` (that follows
    another region's error), else the first error.
    """
    results: dict[str, T] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run(region: str, task: Callable[[], T]) -> None:
        try:
            results[region] = task()
        except BaseException as error:  # re-raised in the calling thread
            with lock:
                errors.append(error)
            transport.abort(tasks)
        finally:
            transport.close(region)

    threads = [
        threading.Thread(
            target=contextvars.copy_context().run, args=(run, region, task), name=region
        )
        for region, task in tasks.items()
    ]
    transport.start(tasks)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise next((e for e in errors if not isinstance(e, TransportClosed)), errors[0])
    return {region: results[region] for region in tasks}
