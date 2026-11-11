# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A region's coupling op: the loop of one coupling node in the region's graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

from neofoam.framework import execution_observer
from neofoam.framework.operations import IterativeOp

from .comm_channel import Data, Tag
from .controller import CouplingController, GroupSide
from .data import End, ExchangeSide, _check_received, _sent
from .kernels import BoundMapping

if TYPE_CHECKING:
    from neofoam.framework.context import Context

    from .region import CouplingNode, Exchange, RegionCoupling


# the loops enclosing a coupling op, outermost first: (name, iteration from 0)
_Loops = tuple[tuple[Optional[str], int], ...]


def _path(names: Sequence[Optional[str]]) -> str:
    """``time_loop > ab``, or ``top``."""
    return " > ".join(str(name) for name in names) or "top"


def _state(loops: _Loops) -> str:
    """``time_loop#3 > ab#1`` (passes from 1), or ``top``."""
    return " > ".join(f"{name}#{iteration + 1}" for name, iteration in loops) or "top"


# the leader's decision for a region: another pass?, final?, by patch the carried
# data or None (receive it fresh), and the patches whose data to send fresh
_Decision = tuple[bool, bool, dict[str, Optional[Data]], frozenset[str]]


class CouplingOp(IterativeOp):
    """A region's coupling node: a loop whose test exchanges the node's sides' data.

    Labelled ``<region>.<node>``. The first test exchanges the sides' models and
    points with the partners; every test sends the leader what the sides send and
    hands each side its mapped data as the leader decides. Not thread safe.
    """

    def __init__(
        self, coupling: RegionCoupling, node: CouplingNode, sides: Sequence[ExchangeSide]
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
        self._by_patch = {
            dict(exchange.ends)[self.region]: exchange
            for exchange in self._exchanges
            if self.region in dict(exchange.ends)
        }
        self._sides = tuple(sides)
        self._check()
        # set at the first test
        self._connected = False
        self._mappers: dict[str, BoundMapping] = {}
        self._controller: Optional[CouplingController] = None
        # the patches whose data the partner receives fresh, as the leader last decided
        self._fresh: frozenset[str] = frozenset()

    def _check(self) -> None:
        patches = [side.patch for side in self._sides]
        for side in self._sides:
            if patches.count(side.patch) > 1:
                raise ValueError(
                    f"two coupled sides on patch {side.label}; an exchange joins patches "
                    "with one coupled side each"
                )
        for patch, exchange in self._by_patch.items():
            if patch not in patches:
                raise ValueError(
                    f"exchange '{exchange.name}' connects {self.region}.{patch}, which has no "
                    "coupled side (no coupled boundary condition on that patch)"
                )

    @property
    def sides(self) -> tuple[ExchangeSide, ...]:
        """The region's sides, in the order they receive."""
        return self._sides

    @property
    def is_leader(self) -> bool:
        """Whether this op decides for its node: its region steps last of the node's."""
        return self.region == self._leader

    @property
    def group(self) -> tuple[str, ...]:
        """The regions the node's exchanges connect with this region, in stepping order."""
        return self._group

    @property
    def controller(self) -> Optional[CouplingController]:
        """The leader's controller, built at its first loop test; ``None`` in other regions."""
        return self._controller

    def exchange_of(self, side: ExchangeSide) -> Exchange:
        """The exchange ``side`` belongs to."""
        return self._by_patch[side.patch]

    def partner(self, side: ExchangeSide) -> str:
        """``<region>.<patch>`` of the side ``side`` receives from."""
        region, patch = self.exchange_of(side).partner(self.region)
        return f"{region}.{patch}"

    def _test(self, ctx: Context) -> bool:
        loops = self._enclosing(ctx)
        iteration = ctx.loops[-1].iteration
        if not self._connected:
            self._connect(loops)
        time = getattr(ctx.time, "value", None)
        current = {side.patch: _sent(side, self.node.name, iteration, time) for side in self._sides}
        for side in self._sides:
            if side.patch in self._fresh:
                exchange = self.exchange_of(side)
                self._send(
                    exchange.partner(self.region)[0], Tag.data(exchange.name), current[side.patch]
                )
        go_on, final, decided, self._fresh = self._decide(
            current, time if iteration == 0 else None, loops, iteration
        )
        if not go_on:
            self._fresh = frozenset()
            return False
        ctx.mark_final(final)
        for side in self._sides:
            self._hand_over(side, decided)
        return True

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
        """Tell each partner its side and the leader the region's sides and loops; check them."""
        path = tuple(name for name, _iteration in loops)
        for side in self._sides:
            exchange = self.exchange_of(side)
            announced = {
                "label": side.label,
                "model": side.sends.__qualname__,
                "fields": tuple(side.sends.model_fields),
                "points": side.coordinates,
            }
            self._send(exchange.partner(self.region)[0], Tag.side(exchange.name), announced)
        sides = [(side.patch, side.field) for side in self._sides]
        if not self.is_leader:
            self._send(self._leader, self._tag("sides"), (path, sides))
        for side in self._sides:
            exchange = self.exchange_of(side)
            announced = self._receive(exchange.partner(self.region)[0], Tag.side(exchange.name))
            _check_received(exchange.name, announced, side)
            try:
                mapper = exchange.mapping.bind(announced["points"], side.coordinates)
            except ValueError as error:
                raise ValueError(
                    f"exchange '{exchange.name}' ({announced['label']} -> {side.label}): {error}"
                ) from None
            self._mappers[side.patch] = mapper
        if self.is_leader:
            self._controller = CouplingController(
                self._node_sides(path, sides), self.label, self.node
            )
        self._connected = True

    def _tag(self, what: Literal["sides", "pass", "decision"]) -> str:
        return Tag.node(self.node.name, what)

    def _node_sides(
        self, path: tuple[Optional[str], ...], own: list[tuple[str, str]]
    ) -> list[GroupSide]:
        """The leader: every region's sides; ``ValueError`` when their loops differ."""
        received = {
            region: (path, own)
            if region == self.region
            else self._receive(region, self._tag("sides"))
            for region in self._group
        }
        paths = {region: there for region, (there, _sides) in received.items()}
        if len(set(paths.values())) > 1:
            where = ", ".join(f"{region}: {_path(there)}" for region, there in paths.items())
            raise ValueError(
                f"coupling '{self.node.name}' sits in different loops in its regions: {where}; "
                "a node must run equally often in all its regions \u2013 put the exchanges in "
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
            GroupSide(region, patch, field, other[(region, patch)], exchange_name[(region, patch)])
            for region, (_path, sides) in received.items()
            for patch, field in sides
        ]

    def _decide(
        self, current: dict[str, Data], time: Any, loops: _Loops, iteration: int
    ) -> _Decision:
        """The leader's decision for this region; the leader checks the regions' times and loops."""
        leader = self._leader
        if not self.is_leader:
            self._send(leader, self._tag("pass"), (time, loops, current))
            decision: _Decision = self._receive(leader, self._tag("decision"))
            return decision
        assert self._controller is not None
        sent = {
            region: (time, loops, current)
            if region == self.region
            else self._receive(region, self._tag("pass"))
            for region in self._group
        }
        times = {region: when for region, (when, _loops, _data) in sent.items()}
        if len(set(times.values())) > 1:
            raise ValueError(
                f"the coupled regions {list(times)} start a time step at different times "
                f"{times}; they must step the same times (deltaT, endTime)"
            )
        states = {region: there for region, (_when, there, _data) in sent.items()}
        if len(set(states.values())) > 1:
            where = ", ".join(f"{region}: {_state(there)}" for region, there in states.items())
            raise ValueError(
                f"coupling '{self.node.name}': its regions entered pass {iteration + 1} "
                f"in different loop states: {where}"
            )
        controller = self._controller
        carried = controller.decide(
            {
                (region, patch): data
                for region, (_when, _loops, sides) in sent.items()
                for patch, data in sides.items()
            },
            iteration,
        )
        received: dict[str, dict[str, Optional[Data]]] = {region: {} for region in self._group}
        fresh: dict[str, set[str]] = {region: set() for region in self._group}
        if carried is not None:
            for end, side in controller.sides.items():
                if controller.fresh(end):
                    received[side.region][side.patch] = None
                    fresh[side.partner[0]].add(side.partner[1])
                else:
                    received[side.region][side.patch] = carried[side.partner]
        decisions = {
            region: (carried is not None, controller.final, by_patch, frozenset(fresh[region]))
            for region, by_patch in received.items()
        }
        for region, decision in decisions.items():
            if region != self.region:
                self._send(region, self._tag("decision"), decision)
        return decisions[self.region]

    def _hand_over(self, side: ExchangeSide, decided: dict[str, Optional[Data]]) -> None:
        """Map what the partner of ``side`` sent (or the carried data) onto it and hand it over."""
        exchange = self.exchange_of(side)
        data = decided[side.patch]
        if data is None:
            data = self._receive(exchange.partner(self.region)[0], Tag.data(exchange.name))
        received = side.receives.model_validate(self._mappers[side.patch](data))
        observer = execution_observer.current()
        if observer is not None:
            observer.coupling_exchanged(self, side, received)
        side.receive(received)
