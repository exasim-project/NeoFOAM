# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A region's coupling op: the loop of one coupling node in the region's graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence

from neofoam.framework import execution_observer
from neofoam.framework.operations import IterativeOp

from ..comm.channel import Data, Tag
from ..data import CoupledPort, End, Stream, check_received, check_rows, sent
from ..location import Geometry
from ..strategies.mapping import BoundMapping
from ._leader import GroupStream, Leader, StreamId

if TYPE_CHECKING:
    from neofoam.framework.context import Context

    from ..exchanges import CouplingLoop, Exchange, RegionCoupling


# the loops enclosing a coupling op, outermost first: (name, iteration from 0)
_Loops = tuple[tuple[Optional[str], int], ...]

# a stream of this region: (port, data key)
_Local = tuple[str, str]


def _path(names: Sequence[Optional[str]]) -> str:
    """``time_loop > ab``, or ``top``."""
    return " > ".join(str(name) for name in names) or "top"


def _state(loops: _Loops) -> str:
    """``time_loop#3 > ab#1`` (passes from 1), or ``top``."""
    return " > ".join(f"{name}#{iteration + 1}" for name, iteration in loops) or "top"


# the leader's decision for a region: another pass?, final?, by received stream the
# carried data or None (receive it fresh), and the sent streams to send fresh
_Decision = tuple[bool, bool, dict[_Local, Optional[Data]], frozenset[_Local]]


class CouplingOp(IterativeOp):
    """A region's coupling node: a loop whose test exchanges the node's ports' data.

    Labelled ``<region>.<node>``. The first test tells each partner what the
    port sends and its geometry and checks the pairing; every test sends the
    leader what the ports send, one stream per data model, and hands each port
    its mapped data as the leader decides. ``geometries``: each port's, by name.
    Not thread safe.
    """

    def __init__(
        self,
        coupling: RegionCoupling,
        node: CouplingLoop,
        ports: Sequence[CoupledPort],
        geometries: Mapping[str, Geometry],
    ) -> None:
        super().__init__(self._test)
        self.region = coupling.region
        self.node = node
        self.label = f"{coupling.region}.{node.name}"
        self._coupling = coupling
        self._comm_channel = coupling.comm_channel
        self._exchanges = coupling.node_exchanges(node)
        self._group = coupling.node_group(node)
        self._leader = self._group[-1]
        self._by_port = {
            dict(exchange.ends)[self.region]: exchange
            for exchange in self._exchanges
            if self.region in dict(exchange.ends)
        }
        self._ports = tuple(ports)
        self._geometries = dict(geometries)
        self._check()
        # set at the first test
        self._connected = False
        self._mappers: dict[str, BoundMapping] = {}
        self._controller: Optional[Leader] = None
        # the streams whose data the partner receives fresh, as the leader last decided
        self._fresh: frozenset[_Local] = frozenset()

    def _check(self) -> None:
        names = [port.name for port in self._ports]
        for name, exchange in self._by_port.items():
            if name not in names:
                raise ValueError(
                    f"exchange '{exchange.name}' connects {self.region}.{name}, which has no "
                    "coupled instance (no coupling(ctx).add on that location)"
                )

    @property
    def ports(self) -> tuple[CoupledPort, ...]:
        """The region's ports, in the order they receive."""
        return self._ports

    @property
    def is_leader(self) -> bool:
        """Whether this op decides for its node: its region steps last of the node's."""
        return self.region == self._leader

    @property
    def group(self) -> tuple[str, ...]:
        """The regions the node's exchanges connect with this region, in stepping order."""
        return self._group

    @property
    def controller(self) -> Optional[Leader]:
        """The leader's controller, built at its first loop test; ``None`` in other regions."""
        return self._controller

    def exchange_of(self, port: CoupledPort) -> Exchange:
        """The exchange ``port`` belongs to."""
        return self._by_port[port.name]

    def partner(self, port: CoupledPort) -> str:
        """``<region>.<port>`` of the port ``port`` exchanges with."""
        region, name = self.exchange_of(port).partner(self.region)
        return f"{region}.{name}"

    def _test(self, ctx: Context) -> bool:
        loops = self._enclosing(ctx)
        iteration = ctx.loops[-1].iteration
        if not self._connected:
            self._connect(loops)
        time = getattr(ctx.time, "value", None)
        current = self._current((iteration, time))
        for port in self._ports:
            exchange = self.exchange_of(port)
            for stream in port.sends:
                if (port.name, stream.key) in self._fresh:
                    tag = Tag.data(exchange.name, stream.key)
                    self._send(self._partner_region(port), tag, current[(port.name, stream.key)])
        go_on, final, decided, self._fresh = self._decide(
            current, time if iteration == 0 else None, loops, iteration
        )
        if not go_on:
            self._fresh = frozenset()
            return False
        ctx.mark_final(final)
        for port in self._ports:
            self._hand_over(port, decided)
        return True

    def _current(self, when: tuple[int, Any]) -> dict[_Local, Data]:
        """What every port sends now, by stream; ``ValueError`` for a wrong row count."""
        current: dict[_Local, Data] = {}
        for port in self._ports:
            size = self._geometries[port.name].size
            for stream in port.sends:
                data = sent(stream, port, self.node.name, when)
                check_rows(stream, data, size, port)
                current[(port.name, stream.key)] = data
        return current

    def _partner_region(self, port: CoupledPort) -> str:
        return self.exchange_of(port).partner(self.region)[0]

    def _enclosing(self, ctx: Context) -> _Loops:
        """``ctx.loops`` without the op's own entry; coupling ops by node name, to compare."""
        nodes = {f"{self.region}.{node.name}": node.name for node in self._coupling.nodes}
        return tuple(
            (nodes.get(entry.name or "", entry.name), entry.iteration) for entry in ctx.loops[:-1]
        )

    def _send(self, target: str, tag: str, message: Any) -> None:
        self._comm_channel.send(self.region, target, tag, message)

    def _receive(self, source: str, tag: str) -> Any:
        return self._comm_channel.receive(source, self.region, tag)

    def _connect(self, loops: _Loops) -> None:
        """Tell each partner what a port sends and the leader the region's streams; check them."""
        path = tuple(name for name, _iteration in loops)
        for port in self._ports:
            announced = {
                "label": port.label,
                "sends": {stream.key: stream.model.__qualname__ for stream in port.sends},
                "entities": {stream.key: stream.model.entity for stream in port.sends},
                "geometry": self._geometries[port.name],
            }
            self._send(self._partner_region(port), Tag.side(self.exchange_of(port).name), announced)
        streams = [
            (port.name, stream.key, stream.names) for port in self._ports for stream in port.sends
        ]
        if not self.is_leader:
            self._send(self._leader, self._tag("sides"), (path, streams))
        for port in self._ports:
            exchange = self.exchange_of(port)
            announced = self._receive(self._partner_region(port), Tag.side(exchange.name))
            check_received(exchange.name, announced, port)
            if port.receives:
                self._mappers[port.name] = self._bind(exchange, announced, port)
        if self.is_leader:
            self._controller = Leader(
                self._node_streams(path, streams), self.label, self.node, self._group
            )
        self._connected = True

    def _bind(self, exchange: Exchange, announced: dict[str, Any], port: CoupledPort) -> Any:
        """The exchange's mapping onto ``port``; ``ValueError`` naming both ports."""
        try:
            for stream in port.receives:
                exchange.mapping.check(announced["entities"][stream.key], stream.model.entity)
            return exchange.mapping.bind(announced["geometry"], self._geometries[port.name])
        except ValueError as error:
            raise ValueError(
                f"exchange '{exchange.name}' ({announced['label']} -> {port.label}): {error}"
            ) from None

    def _tag(self, what: Literal["sides", "pass", "decision"]) -> str:
        return Tag.node(self.node.name, what)

    def _node_streams(self, path: tuple[Optional[str], ...], own: list[Any]) -> list[GroupStream]:
        """The leader: every region's streams; ``ValueError`` when their loops differ."""
        received = {
            region: (path, own)
            if region == self.region
            else self._receive(region, self._tag("sides"))
            for region in self._group
        }
        paths = {region: there for region, (there, _streams) in received.items()}
        if len(set(paths.values())) > 1:
            where = ", ".join(f"{region}: {_path(there)}" for region, there in paths.items())
            raise ValueError(
                f"coupling '{self.node.name}' sits in different loops in its regions: {where}; "
                "a node must run equally often in all its regions – put the exchanges in "
                "one node or place it alike"
            )
        other: dict[End, End] = {
            end: exchange.ends[1 - k]
            for exchange in self._exchanges
            for k, end in enumerate(exchange.ends)
        }
        exchange_name = {
            end: exchange.name for exchange in self._exchanges for end in exchange.ends
        }
        return [
            GroupStream(
                region,
                port,
                key,
                tuple(names),
                other[(region, port)],
                exchange_name[(region, port)],
            )
            for region, (_path, streams) in received.items()
            for port, key, names in streams
        ]

    def _decide(
        self, current: dict[_Local, Data], time: Any, loops: _Loops, iteration: int
    ) -> _Decision:
        """The leader's decision for this region; the leader checks the regions' times and loops."""
        if not self.is_leader:
            self._send(self._leader, self._tag("pass"), (time, loops, current))
            decision: _Decision = self._receive(self._leader, self._tag("decision"))
            return decision
        sent_by = {
            region: (time, loops, current)
            if region == self.region
            else self._receive(region, self._tag("pass"))
            for region in self._group
        }
        self._check_pass(sent_by, iteration)
        decisions = self._decisions(sent_by, iteration)
        for region, decision in decisions.items():
            if region != self.region:
                self._send(region, self._tag("decision"), decision)
        return decisions[self.region]

    def _check_pass(self, sent_by: dict[str, Any], iteration: int) -> None:
        """Raise ``ValueError`` unless the regions entered the pass at one time and loop state."""
        times = {region: when for region, (when, _loops, _data) in sent_by.items()}
        if len(set(times.values())) > 1:
            raise ValueError(
                f"the coupled regions {list(times)} start a time step at different times "
                f"{times}; they must step the same times (deltaT, endTime)"
            )
        states = {region: there for region, (_when, there, _data) in sent_by.items()}
        if len(set(states.values())) > 1:
            where = ", ".join(f"{region}: {_state(there)}" for region, there in states.items())
            raise ValueError(
                f"coupling '{self.node.name}': its regions entered pass {iteration + 1} "
                f"in different loop states: {where}"
            )

    def _decisions(self, sent_by: dict[str, Any], iteration: int) -> dict[str, _Decision]:
        """The controller's decision, for each region: what it receives and sends fresh."""
        controller = self._controller
        assert controller is not None
        current: dict[StreamId, Data] = {
            (region, port, key): data
            for region, (_when, _loops, streams) in sent_by.items()
            for (port, key), data in streams.items()
        }
        carried = controller.decide(current, iteration)
        received: dict[str, dict[_Local, Optional[Data]]] = {region: {} for region in self._group}
        fresh: dict[str, set[_Local]] = {region: set() for region in self._group}
        if carried is not None:
            for stream_id, stream in controller.streams.items():
                partner_region, partner_port = stream.partner
                into = received[partner_region]
                if controller.fresh(stream_id):
                    into[(partner_port, stream.key)] = None
                    fresh[stream.region].add((stream.port, stream.key))
                else:
                    into[(partner_port, stream.key)] = carried[stream_id]
        return {
            region: (carried is not None, controller.final, by_stream, frozenset(fresh[region]))
            for region, by_stream in received.items()
        }

    def _hand_over(self, port: CoupledPort, decided: dict[_Local, Optional[Data]]) -> None:
        """Map what the partner sent (or the carried data) onto ``port``; hand each stream over."""
        for stream in port.receives:
            received = self._mapped(port, stream, decided[(port.name, stream.key)])
            observer = execution_observer.current()
            if observer is not None:
                observer.coupling_exchanged(self, port, received)
            stream.method(received)

    def _mapped(self, port: CoupledPort, stream: Stream, data: Optional[Data]) -> Any:
        if data is None:
            tag = Tag.data(self.exchange_of(port).name, stream.key)
            data = self._receive(self._partner_region(port), tag)
        return stream.model.model_validate(self._mappers[port.name](data))
