# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What one region knows of the coupling of a case: :class:`RegionCoupling`.

Regions that exchanges connect form a group; its last region in stepping order
leads it (agrees the time step). A node's last region leads the node (decides
when it converged). A serial pass waits for the whole pass of the regions that
step before it, so a serial node may neither enclose a node coupling two of its
regions nor hold their time step: both need the partner back inside the pass.

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
from typing import Callable, Literal, Mapping, Sequence, TypeVar

from neofoam.framework.types import Placement

from . import kernels
from .data import End
from .kernels import CouplingAcceleration, CouplingConvergence
from .transport import Transport, TransportClosed

T = TypeVar("T")


@dataclass(frozen=True)
class Link:
    """One exchange of a case: its unique name, its two ends ``(region, patch)``, its mapping.

    Nearest-neighbor by default, which needs the coordinates of both patches;
    :class:`~neofoam.framework.coupling.IdentityMapping` for patches sharing their points.
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
    """A named coupling loop over some exchanges; one op ``<region>.<name>`` per region.

    Once every measure of ``convergence`` holds (none: the data is unchanged) one
    more pass runs, the final one; the last of ``max_iterations`` is final
    regardless. ``acceleration`` relaxes the carried data, by exchange name;
    ``placement`` places the op per region (default: around the whole top scope).
    ``serial`` (the default): a side receives what an earlier-stepping partner sent
    in the same pass; ``parallel``: what it sent in the pass before.

    An acceleration keeps state across passes: give each node its own.
    """

    name: str
    exchanges: tuple[str, ...]
    max_iterations: int = 100
    convergence: tuple[CouplingConvergence, ...] = ()
    acceleration: Mapping[str, CouplingAcceleration] = field(default_factory=dict)
    placement: Mapping[str, Placement] = field(default_factory=dict)
    scheme: Literal["serial", "parallel"] = "serial"

    @property
    def parallel(self) -> bool:
        """Whether the node's scheme is ``parallel``."""
        return self.scheme == "parallel"

    def placement_of(self, region: str) -> Placement:
        """Where the node's op sits in the graph of ``region``."""
        return self.placement.get(region, Placement())


@dataclass(frozen=True)
class RegionCoupling:
    """What region ``region`` knows of its case's coupling, built by :func:`region_couplings`.

    ``group``: the regions coupled with it in stepping order; ``links``: the
    group's exchanges; ``nodes``: those with an exchange of this region, outer to inner.
    """

    region: str
    group: tuple[str, ...]
    links: tuple[Link, ...]
    transport: Transport
    nodes: tuple[CouplingNode, ...] = ()

    @property
    def leader(self) -> str:
        """The group's last region in stepping order; it agrees the time step."""
        return self.group[-1]

    @property
    def is_leader(self) -> bool:
        """Whether this region leads its group."""
        return self.region == self.leader

    @property
    def patches(self) -> tuple[str, ...]:
        """The patches of this region that an exchange connects."""
        return tuple(
            dict(link.ends)[self.region] for link in self.links if self.region in dict(link.ends)
        )

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


def region_couplings(
    regions: Sequence[str],
    links: Sequence[Link],
    transport: Transport,
    nodes: Sequence[CouplingNode],
) -> dict[str, RegionCoupling]:
    """The coupling of every region ``links`` connect, by region; ``regions`` in stepping order.

    ``nodes`` lists the coupling nodes, outer to inner. Raises ``ValueError`` as
    :func:`check_couplings`.
    """
    check_couplings(regions, links, nodes)
    couplings: dict[str, RegionCoupling] = {}
    for region in regions:
        group = _group(region, regions, links)
        if region in couplings or len(group) < 2:
            continue
        group_links = tuple(link for link in links if link.ends[0][0] in group)
        for member in group:
            touching = {link.name for link in group_links if member in dict(link.ends)}
            member_nodes = tuple(node for node in nodes if touching & set(node.exchanges))
            couplings[member] = RegionCoupling(member, group, group_links, transport, member_nodes)
    return couplings


def check_couplings(
    regions: Sequence[str], links: Sequence[Link], nodes: Sequence[CouplingNode]
) -> None:
    """Raise ``ValueError`` unless ``links`` and ``nodes`` couple ``regions`` consistently.

    The one check of a case's coupling, for the Python API and the case's config:
    every link joins two regions of ``regions`` on patches no other link has and
    has a name of its own; every exchange is in exactly one node; a node's
    placement and acceleration name only its own regions and exchanges; and no
    serial node encloses a node that couples two of its regions as well.
    """
    _check_links(regions, links)
    names = [link.name for link in links]
    in_node: dict[str, str] = {}
    for node in nodes:
        if node.name in [other.name for other in nodes if other is not node]:
            raise ValueError(f"two coupling nodes are named '{node.name}'; a node's name is unique")
        _check_node(node, links)
        for exchange in node.exchanges:
            if exchange in in_node:
                raise ValueError(
                    f"exchange '{exchange}' is in the coupling nodes '{in_node[exchange]}' and "
                    f"'{node.name}'; an exchange is in exactly one node"
                )
            in_node[exchange] = node.name
    missing = [name for name in names if name not in in_node]
    if missing:
        raise ValueError(
            f"the exchanges {missing} are in no coupling node; each exchange is in exactly one "
            f"node (the nodes: {[node.name for node in nodes]})"
        )
    _check_serial_nesting(nodes, links)


def _check_links(regions: Sequence[str], links: Sequence[Link]) -> None:
    """Raise ``ValueError`` unless each link joins two known regions on patches of its own."""
    named = sorted({region for link in links for region, _patch in link.ends} - set(regions))
    if named:
        raise ValueError(
            f"the exchanges couple the regions {named}, which are not in {list(regions)}"
        )
    names = [link.name for link in links]
    owner: dict[End, str] = {}
    for link in links:
        if names.count(link.name) > 1:
            raise ValueError(f"two exchanges are named '{link.name}'; an exchange's name is unique")
        first, second = link.ends
        if first[0] == second[0]:
            raise ValueError(f"exchange '{link.name}' joins two patches of region {first[0]}")
        for end in link.ends:
            if end in owner:
                raise ValueError(
                    f"{end[0]}.{end[1]} is in the exchanges '{owner[end]}' and "
                    f"'{link.name}'; a coupled patch is in one exchange"
                )
            owner[end] = link.name


def _check_node(node: CouplingNode, links: Sequence[Link]) -> None:
    """Raise ``ValueError`` unless ``node`` names existing exchanges and only its own regions."""
    names = [link.name for link in links]
    if not node.exchanges:
        raise ValueError(f"coupling node '{node.name}' has no exchange")
    unknown_exchanges = [exchange for exchange in node.exchanges if exchange not in names]
    if unknown_exchanges:
        raise ValueError(
            f"coupling node '{node.name}' names the exchange '{unknown_exchanges[0]}', which "
            f"does not exist; the exchanges are {names}"
        )
    regions = sorted(_regions(node, links))
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


def _regions(node: CouplingNode, links: Sequence[Link]) -> set[str]:
    """The regions the exchanges of ``node`` connect."""
    return {region for link in links if link.name in node.exchanges for region, _ in link.ends}


def _check_serial_nesting(nodes: Sequence[CouplingNode], links: Sequence[Link]) -> None:
    """Raise ``ValueError`` when a serial node encloses a node sharing two of its regions."""
    for index, outer in enumerate(nodes):
        if outer.parallel:
            continue
        for inner in nodes[index + 1 :]:
            shared = sorted(_regions(outer, links) & _regions(inner, links))
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
    """Run every region's task on its own thread; each task's result, by region.

    The threads report to the calling thread's execution observer. A failing
    region aborts the transport for all, so none waits forever; then the first
    error that is not a :class:`TransportClosed` (a consequence) is raised.
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
