# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What one region knows of the coupling of a case: :class:`ParticipantExchanges`.

A serial pass waits for the whole pass of the regions that step before it, so a
serial node may neither enclose a node coupling two of its regions nor hold their
time step: both need the partner back inside the pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from neofoam.framework.types import Placement

from .comm.channel import CommChannel
from .data import End
from .strategies.acceleration import CouplingAcceleration
from .strategies.convergence import CouplingConvergence
from .strategies.mapping import CouplingMapping, default_mapping


@dataclass(frozen=True)
class Exchange:
    """One exchange of a case: its unique name, its two ends ``(region, port)``, its mapping.

    The exchange pairs what one port sends with what the other receives, by data
    model. Nearest-neighbor by default, which needs the points of both ports;
    :class:`~neofoam.framework.coupling.IdentityMapping` for ports sharing their entities.
    """

    name: str
    ends: tuple[End, End]
    mapping: CouplingMapping = field(default_factory=default_mapping)

    def partner(self, region: str) -> End:
        """The end of the exchange that is not in ``region``."""
        first, second = self.ends
        return second if first[0] == region else first


@dataclass(frozen=True)
class CouplingLoop:
    """A named coupling loop over some exchanges; one op ``<region>.<name>`` per region.

    Once every measure of ``convergence`` holds (none: the data is unchanged) one
    more pass runs, the final one; the last of ``max_iterations`` is final
    regardless. ``acceleration`` relaxes the carried data, by exchange name, and
    keeps state across passes: give each node its own. ``placement`` places the op
    per region (default: around the whole top scope). ``serial``: a side receives
    what an earlier-stepping partner sent in the same pass; ``parallel``: what it
    sent in the pass before.
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
        return self.scheme == "parallel"

    def placement_of(self, region: str) -> Placement:
        """Where the node's op sits in the graph of ``region``."""
        return self.placement.get(region, Placement())


@dataclass(frozen=True)
class ParticipantExchanges:
    """What region ``region`` knows of its case's coupling, built by :func:`participant_exchanges`.

    ``group``: the regions coupled with it in stepping order; ``exchanges``: the
    group's exchanges; ``nodes``: those with an exchange of this region, outer to inner.
    """

    region: str
    group: tuple[str, ...]
    exchanges: tuple[Exchange, ...]
    comm_channel: CommChannel
    nodes: tuple[CouplingLoop, ...] = ()

    @property
    def leader(self) -> str:
        """The group's last region in stepping order; it agrees the time step."""
        return self.group[-1]

    @property
    def is_leader(self) -> bool:
        """Whether this region leads its group."""
        return self.region == self.leader

    @property
    def ports(self) -> tuple[str, ...]:
        """The ports of this region that an exchange connects."""
        return tuple(
            dict(exchange.ends)[self.region]
            for exchange in self.exchanges
            if self.region in dict(exchange.ends)
        )

    def node_exchanges(self, node: CouplingLoop) -> tuple[Exchange, ...]:
        """The exchanges of ``node`` in this region's group."""
        return tuple(exchange for exchange in self.exchanges if exchange.name in node.exchanges)

    def node_of(self, port: str) -> CouplingLoop:
        """The node whose exchange connects ``port`` of this region."""
        for node in self.nodes:
            for exchange in self.node_exchanges(node):
                if dict(exchange.ends).get(self.region) == port:
                    return node
        raise KeyError(f"no coupling node has an exchange on {self.region}.{port}")

    def node_group(self, node: CouplingLoop) -> tuple[str, ...]:
        """The regions ``node`` connects with this region, itself included, in stepping order."""
        return _group(self.region, self.group, self.node_exchanges(node))


def participant_exchanges(
    regions: Sequence[str],
    exchanges: Sequence[Exchange],
    comm_channel: CommChannel,
    nodes: Sequence[CouplingLoop],
) -> dict[str, ParticipantExchanges]:
    """The coupling of every region ``exchanges`` connect, by region.

    ``regions`` are in stepping order, ``nodes`` outer to inner.
    """
    check_exchanges(regions, exchanges, nodes)
    couplings: dict[str, ParticipantExchanges] = {}
    for region in regions:
        group = _group(region, regions, exchanges)
        if region in couplings or len(group) < 2:
            continue
        group_exchanges = tuple(exchange for exchange in exchanges if exchange.ends[0][0] in group)
        for member in group:
            touching = {
                exchange.name for exchange in group_exchanges if member in dict(exchange.ends)
            }
            member_nodes = tuple(node for node in nodes if touching & set(node.exchanges))
            couplings[member] = ParticipantExchanges(
                member, group, group_exchanges, comm_channel, member_nodes
            )
    return couplings


def check_exchanges(
    regions: Sequence[str], exchanges: Sequence[Exchange], nodes: Sequence[CouplingLoop]
) -> None:
    """Raise ``ValueError`` unless ``exchanges`` and ``nodes`` couple ``regions`` consistently.

    The one check of a case's coupling, for the Python API and the case's config.
    """
    _check_ends(regions, exchanges)
    names = [exchange.name for exchange in exchanges]
    in_node: dict[str, str] = {}
    for node in nodes:
        if node.name in [other.name for other in nodes if other is not node]:
            raise ValueError(f"two coupling nodes are named '{node.name}'; a node's name is unique")
        _check_node(node, exchanges)
        for exchange_name in node.exchanges:
            if exchange_name in in_node:
                raise ValueError(
                    f"exchange '{exchange_name}' is in the coupling nodes "
                    f"'{in_node[exchange_name]}' and '{node.name}'; an exchange is in exactly "
                    "one node"
                )
            in_node[exchange_name] = node.name
    missing = [name for name in names if name not in in_node]
    if missing:
        raise ValueError(
            f"the exchanges {missing} are in no coupling node; each exchange is in exactly one "
            f"node (the nodes: {[node.name for node in nodes]})"
        )
    _check_serial_nesting(nodes, exchanges)


def _check_ends(regions: Sequence[str], exchanges: Sequence[Exchange]) -> None:
    named = sorted(
        {region for exchange in exchanges for region, _port in exchange.ends} - set(regions)
    )
    if named:
        raise ValueError(
            f"the exchanges couple the regions {named}, which are not in {list(regions)}"
        )
    names = [exchange.name for exchange in exchanges]
    owner: dict[End, str] = {}
    for exchange in exchanges:
        if names.count(exchange.name) > 1:
            raise ValueError(
                f"two exchanges are named '{exchange.name}'; an exchange's name is unique"
            )
        first, second = exchange.ends
        if first[0] == second[0]:
            raise ValueError(f"exchange '{exchange.name}' joins two ports of region {first[0]}")
        for end in exchange.ends:
            if end in owner:
                raise ValueError(
                    f"{end[0]}.{end[1]} is in the exchanges '{owner[end]}' and "
                    f"'{exchange.name}'; a coupled port is in one exchange"
                )
            owner[end] = exchange.name


def _check_node(node: CouplingLoop, exchanges: Sequence[Exchange]) -> None:
    names = [exchange.name for exchange in exchanges]
    if not node.exchanges:
        raise ValueError(f"coupling node '{node.name}' has no exchange")
    unknown_exchanges = [name for name in node.exchanges if name not in names]
    if unknown_exchanges:
        raise ValueError(
            f"coupling node '{node.name}' names the exchange '{unknown_exchanges[0]}', which "
            f"does not exist; the exchanges are {names}"
        )
    regions = sorted(_regions(node, exchanges))
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


def _regions(node: CouplingLoop, exchanges: Sequence[Exchange]) -> set[str]:
    return {
        region
        for exchange in exchanges
        if exchange.name in node.exchanges
        for region, _ in exchange.ends
    }


def _check_serial_nesting(nodes: Sequence[CouplingLoop], exchanges: Sequence[Exchange]) -> None:
    for index, outer in enumerate(nodes):
        if outer.parallel:
            continue
        for inner in nodes[index + 1 :]:
            shared = sorted(_regions(outer, exchanges) & _regions(inner, exchanges))
            if len(shared) >= 2:
                raise ValueError(
                    f"coupling '{outer.name}' is serial but encloses '{inner.name}', which "
                    f"couples {shared} as well; a serial outer pass waits for its partner's "
                    "whole pass, which needs the inner coupling of both - make "
                    f"'{outer.name}' parallel"
                )


def _group(region: str, regions: Sequence[str], exchanges: Sequence[Exchange]) -> tuple[str, ...]:
    """The regions ``exchanges`` connect with ``region``, itself included, in stepping order."""
    group, todo = {region}, [region]
    while todo:
        current = todo.pop()
        for exchange in exchanges:
            ends = {name for name, _port in exchange.ends}
            if current in ends:
                todo += sorted(ends - group)
                group |= ends
    return tuple(name for name in regions if name in group)
