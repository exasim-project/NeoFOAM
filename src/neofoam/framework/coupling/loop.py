# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The coupling of a region: its coupling op.

Vocabulary:

- a **coupled side** (:class:`~neofoam.framework.coupling.data.ExchangeSide`) is
  what a boundary condition declares on its patch: what it sends and receives
  (one pydantic model each), and its points; an **exchange**
  (:class:`~neofoam.framework.coupling.data.Exchange`) joins it with the side of
  another region (its partner), its mapping maps the partner's data onto the
  side's points (:mod:`~neofoam.framework.coupling.data`);
- a **coupling node** (:class:`~neofoam.framework.coupling.CouplingNode`) is a
  named coupling loop over some exchanges; a region's **coupling op**
  (:class:`CouplingOp`, ``<region>.<node>``) is the node's loop in the region:
  it wraps where the node's placement says (by default the whole body of the
  top scope, or of the node before), and the resolver nests the operations that
  solve them inside it. At the start of each pass its loop test hands every side
  of the region in the node what its partner sent, mapped onto the side's points;
- the **leader** of a node, the last region in stepping order of the regions
  the node's exchanges connect, owns the node's **controller**
  (:class:`~neofoam.framework.coupling.controller.CouplingController`), which
  decides after each pass whether the node's data converged and which pass is
  the node's **final** one. Every region runs the same passes, the final one
  marked final on its loop entry (``LoopEntry.final``), and leaves the node's
  loop after it;
- all data between regions goes through the region's
  :class:`~neofoam.framework.coupling.transport.Transport`: no region reads
  another region's objects.

A boundary condition declares its side with :func:`~neofoam.framework.coupling.coupled`,
which adds itself to its region's coupling model
(:class:`~neofoam.framework.coupling.Coupling`); the model builds the coupling op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

from neofoam.framework import execution_observer
from neofoam.framework.operations import IterativeOp

from .controller import CouplingController, GroupSide
from .data import End, Exchange, ExchangeSide, _array_names, _arrays, _check_received, _kinds, _sent
from .kernels import BoundMapping
from .transport import Data, Tag

if TYPE_CHECKING:
    from neofoam.framework.context import Context

    from .region import CouplingNode, RegionCoupling


# The loops enclosing a region's coupling op, outermost first, as (name, iteration):
# a coupling op by its node's name, other loops by their names; iteration from 0.
_Loops = tuple[tuple[Optional[str], int], ...]


def _path(names: Sequence[Optional[str]]) -> str:
    """``time_loop > ab``: the names of enclosing loops, ``top`` for none."""
    return " > ".join(str(name) for name in names) or "top"


def _state(loops: _Loops) -> str:
    """``time_loop#3 > ab#1``: ``loops`` with their passes (from 1), ``top`` for none."""
    return " > ".join(f"{name}#{iteration + 1}" for name, iteration in loops) or "top"


# The leader's decision for one region: another pass?, is it the final one?, what
# each of its sides receives, by patch: the carried data, or None: what its partner
# sends in this pass; and the patches whose partner receives that (the region sends
# their data at its next loop test, after the pass).
_Decision = tuple[bool, bool, dict[str, Optional[Data]], frozenset[str]]


class CouplingOp(IterativeOp):
    """A region's coupling node: a loop whose test exchanges the node's sides' data, per pass.

    Built from what the region knows of the case's coupling (``coupling``: its
    exchanges, the group's regions, its transport), the ``node`` (its exchanges)
    and the region's coupled ``sides`` of the node (the region's
    :class:`~neofoam.framework.coupling.Coupling` builds it, each side on an
    exchange of the node); raises ``ValueError`` when a patch has two sides, or
    an exchange of the node in the region none. The resolver places it
    as the node's placement for the region says (``metadata.wraps``). Its label
    is ``<region>.<node>``; its messages to other regions are tagged by node
    (``<node>/sides``, ``<node>/pass``, ``<node>/decision``) or by exchange
    (``side:<exchange>``, ``data:<exchange>``).

    A node runs equally often in all its regions, so it sits in the same loops
    in each: the first test tells the leader the names of the loops enclosing
    the op (``Context.loops``, a coupling op by its node's name), and every
    test their passes; the leader raises ``ValueError`` when they differ.

    Its loop test runs before each pass. The first test ever exchanges the
    sides' models and points with the partner regions and checks each exchange.
    Each test then sends what every side sends now to the node's leader, and to
    the partner of every side the leader's decision before named (the partner
    receives it fresh, in the pass of that decision); the leader decides for the
    node and tells every region to go on or stop, and whether the pass is the
    node's final one, which the test marks on the loop's entry
    (``Context.mark_final``: an operation marked ``final`` runs in it). On
    another pass every side gets what the leader decided: what its partner sends
    at its next test (fresh), or what the leader carried, mapped onto its points
    with the exchange's mapping, validated as its ``receives`` model, reported to
    the execution observer (``coupling_exchanged``) and handed to its
    ``receive``. Not thread safe.
    """

    def __init__(
        self, coupling: RegionCoupling, node: CouplingNode, sides: Sequence[ExchangeSide]
    ) -> None:
        super().__init__(self._test)
        #: The region of the op.
        self.region = coupling.region
        #: The coupling node of the op.
        self.node = node
        #: The name of the operation: ``<region>.<node>``.
        self.label = f"{coupling.region}.{node.name}"
        self._coupling = coupling
        self._transport = coupling.transport
        self._links = coupling.node_links(node)
        # the regions of the node, in stepping order; the last one leads it
        self._group = coupling.node_group(node)
        self._leader = self._group[-1]
        self._exchanges = {
            dict(link.ends)[self.region]: Exchange(
                link.name, link.partner(self.region), link.mapping
            )
            for link in self._links
            if self.region in dict(link.ends)
        }
        self._sides = tuple(sides)
        self._check()
        # set at the first test: the bound mapping per patch, and the leader's controller
        self._connected = False
        self._mappers: dict[str, BoundMapping] = {}
        self._controller: Optional[CouplingController] = None
        self._iteration = 0
        # the patches whose data the partner receives fresh: the leader's last decision
        self._fresh: frozenset[str] = frozenset()

    def _check(self) -> None:
        """Raise ``ValueError`` unless every exchange of the node has exactly one side.

        That every side has an exchange of the node :meth:`Coupling.add
        <neofoam.framework.coupling.Coupling.add>` checked, and
        :meth:`Coupling.operations <neofoam.framework.coupling.Coupling.operations>`
        hands each node the sides on its exchanges' patches.
        """
        patches = [side.patch for side in self._sides]
        for side in self._sides:
            if patches.count(side.patch) > 1:
                raise ValueError(
                    f"two coupled sides on patch {side.label}; an exchange joins patches "
                    "with one coupled side each"
                )
        for patch, exchange in self._exchanges.items():
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
        """Whether this coupling op decides for its node (its region steps last of the node's)."""
        return self.region == self._leader

    @property
    def leader(self) -> str:
        """The region that decides for the node: the last of its regions in stepping order."""
        return self._leader

    @property
    def group(self) -> tuple[str, ...]:
        """The regions the node's exchanges connect with this region, in stepping order."""
        return self._group

    @property
    def controller(self) -> Optional[CouplingController]:
        """The leader's controller, built at its first loop test; ``None`` elsewhere."""
        return self._controller

    def exchange_of(self, side: ExchangeSide) -> Exchange:
        """The exchange ``side`` belongs to."""
        return self._exchanges[side.patch]

    def partner(self, side: ExchangeSide) -> str:
        """``<region>.<patch>`` of the side ``side`` receives from."""
        region, patch = self.exchange_of(side).partner
        return f"{region}.{patch}"

    # --- the loop test ------------------------------------------------------------------------

    def _test(self, ctx: Context) -> bool:
        loops = self._enclosing(ctx)
        if not self._connected:
            self._connect(loops)
        time = getattr(ctx.time, "value", None)
        current = {
            side.patch: _sent(side, self.node.name, self._iteration, time) for side in self._sides
        }
        for side in self._sides:
            if side.patch in self._fresh:
                exchange = self.exchange_of(side)
                self._send(exchange.partner[0], Tag.data(exchange.name), current[side.patch])
        go_on, final, decided, self._fresh = self._decide(
            current, time if self._iteration == 0 else None, loops
        )
        if not go_on:
            self._iteration = 0
            self._fresh = frozenset()
            return False
        ctx.mark_final(final)
        self._iteration += 1
        for side in self._sides:
            self._hand_over(side, decided)
        return True

    def _enclosing(self, ctx: Context) -> _Loops:
        """The loops enclosing the op, outermost first: ``ctx.loops`` without its own entry.

        The op runs its test with its own entry last on ``ctx.loops``. A
        coupling op of the region (``<region>.<node>``) is named by its node, so
        that the regions' loops compare.
        """
        nodes = {f"{self.region}.{node.name}": node.name for node in self._coupling.nodes}
        return tuple(
            (nodes.get(entry.name or "", entry.name), entry.iteration) for entry in ctx.loops[:-1]
        )

    def _send(self, target: str, tag: str, message: Any) -> None:
        self._transport.send(self.region, target, tag, message)

    def _receive(self, source: str, tag: str) -> Any:
        return self._transport.receive(source, self.region, tag)

    def _connect(self, loops: _Loops) -> None:
        """Tell each partner its side (models, points), the leader the region's sides; check.

        The leader also gets the names of the ``loops`` enclosing the op. Raises
        ``ValueError`` when a side does not receive what its partner sends or its
        mapping cannot map between their points; in the leader, when the regions'
        ops sit in different loops or the controller cannot be built.
        """
        path = tuple(name for name, _iteration in loops)
        for side in self._sides:
            exchange = self.exchange_of(side)
            announced = {
                "label": side.label,
                "model": side.sends.__qualname__,
                "kinds": _kinds(side.sends),
                "points": side.coordinates,
            }
            self._send(exchange.partner[0], Tag.side(exchange.name), announced)
        sides = [(side.patch, side.field, _array_names(side.sends)) for side in self._sides]
        if not self.is_leader:
            self._send(self._leader, self._tag("sides"), (path, sides))
        for side in self._sides:
            exchange = self.exchange_of(side)
            announced = self._receive(exchange.partner[0], Tag.side(exchange.name))
            _check_received(exchange.name, announced, side)
            self._mappers[side.patch] = exchange.mapping.bind(announced["points"], side.coordinates)
        if self.is_leader:
            self._controller = self._coupling.controller(
                self._node_sides(path, sides),
                self.label,
                node=self.node,
            )
        self._connected = True

    def _tag(self, what: Literal["sides", "pass", "decision"]) -> str:
        """The transport tag ``<node>/<what>`` of the node's messages (:meth:`Tag.node`)."""
        return Tag.node(self.node.name, what)

    def _node_sides(
        self, path: tuple[Optional[str], ...], own: list[tuple[str, str, tuple[str, ...]]]
    ) -> list[GroupSide]:
        """The leader: the sides of every region of the node, as the regions told it.

        Raises ``ValueError`` when the regions' ops sit in different loops (``path``:
        the names of the loops enclosing the leader's).
        """
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
            end: link.ends[1 - k] for link in self._links for k, end in enumerate(link.ends)
        }
        exchange = {end: link.name for link in self._links for end in link.ends}
        return [
            GroupSide(
                region, patch, field, arrays, other[(region, patch)], exchange[(region, patch)]
            )
            for region, (_path, sides) in received.items()
            for patch, field, arrays in sides
        ]

    def _decide(self, current: dict[str, Data], time: Any, loops: _Loops) -> _Decision:
        """The node leader's decision for this region, from what its sides send now (by patch).

        ``time`` is the region's time as a time step's coupling starts, else
        ``None``; ``loops`` the loops enclosing the op, with their passes. The
        leader raises ``ValueError`` when the regions' times or loops' passes differ.
        """
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
                f"coupling '{self.node.name}': its regions entered pass {self._iteration + 1} "
                f"in different loop states: {where}"
            )
        controller = self._controller
        carried = controller.decide(
            {
                (region, patch): data
                for region, (_when, _loops, sides) in sent.items()
                for patch, data in sides.items()
            }
        )
        received: dict[str, dict[str, Optional[Data]]] = {region: {} for region in self._group}
        # the patches of each region whose partner receives their data fresh
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
        """Map what the partner of ``side`` sent onto its points and hand it over.

        ``decided`` is the leader's decision: the carried data by patch, or
        ``None`` where the partner's data of this pass is to be received.
        """
        exchange = self.exchange_of(side)
        data = decided[side.patch]
        if data is None:
            data = self._receive(exchange.partner[0], Tag.data(exchange.name))
        mapped = self._mappers[side.patch](_arrays(side.mapped, data))
        received = side.receives.model_validate({**data, **mapped})
        observer = execution_observer.current()
        if observer is not None:
            observer.coupling_exchanged(self, side, received)
        side.receive(received)
