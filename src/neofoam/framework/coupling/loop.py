# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The coupling of a region: its coupling op and, in the group's leader, the controller.

Vocabulary:

- a **coupled side** (:class:`CoupledSide`) is what a boundary condition declares
  on its patch: what it sends and receives (one pydantic model each), and its points;
- an **exchange** (:class:`Exchange`) joins a side with the side of another
  region (its partner); the case configures it per patch pair; its **mapping**
  (:class:`~neofoam.framework.coupling.kernels.Mapping`) maps the :data:`Array`
  data the partner sends onto the side's points;
- a **coupling node** (:class:`~neofoam.framework.coupling.CouplingNode`) is a
  named coupling loop over some exchanges; a region's **coupling op**
  (:class:`CouplingOp`, ``<region>.<node>``) is the node's loop in the region:
  it wraps where the node's placement says (by default the whole body of the
  top scope, or of the node before), and the resolver nests the operations that
  solve them inside it. At the start of each pass its loop test hands every side
  of the region in the node what its partner sent, mapped onto the side's points;
- the **leader** of a node, the last region in stepping order of the regions
  the node's exchanges connect, owns the node's **controller**
  (:class:`CouplingController`), which decides after each pass whether the
  node's data converged, and relaxes the data carried into the next pass, per
  exchange. It also decides which pass is the node's **final** one: the pass
  after the one its data converged in, or the last one ``max_iterations``
  allows. Every region runs the same passes, the final one marked final on its
  loop entry (``LoopEntry.final``), and leaves the node's loop after it;
- all data between regions goes through the region's
  :class:`~neofoam.framework.coupling.transport.Transport`: no region reads
  another region's objects.

A boundary condition declares its side with :func:`~neofoam.framework.coupling.coupled`,
which adds itself to its region's coupling model
(:class:`~neofoam.framework.coupling.Coupling`); the model builds the coupling op.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Callable, Optional, Sequence

import numpy as np
from pydantic import BaseModel, PlainValidator

from neofoam.framework import execution_observer
from neofoam.framework.operations import IterativeOp

from .kernels import Acceleration, Convergence, Mapping, SideChange, mapping
from .transport import Data, SideKey

if TYPE_CHECKING:
    from neofoam.framework.context import Context

    from .region import CouplingNode, End, RegionCoupling


@dataclass(frozen=True)
class _ArrayKind:
    """Marks a field of an exchange model as :data:`Array`."""


def _as_array(value: Any) -> Any:
    return np.asarray(value, dtype=float)


# A field of an exchange model with one value (or row) per point of the patch; the
# mapping maps it onto the receiving side's points.
Array = Annotated[np.ndarray, _ArrayKind(), PlainValidator(_as_array)]


def _is_model(model: Any) -> bool:
    return isinstance(model, type) and issubclass(model, BaseModel)


@dataclass(frozen=True)
class CoupledSide:
    """One side of an exchange: what it sends and receives, and where its points are.

    ``sends`` and ``receives`` are pydantic models: :data:`Array` fields (one
    value per point of the patch, mapped onto the partner's points) and plain
    fields (one value for the interface, handed over as they are). The same model
    both ways is the special case of a symmetric exchange. ``send()`` returns an
    instance of ``sends``; ``receive(data)`` takes an instance of ``receives``,
    already mapped onto this side's points. ``coordinates`` holds one row per
    point of the patch, or ``None``.

    A boundary condition usually declares it with
    :func:`~neofoam.framework.coupling.coupled` instead of building it.

    Example::

        CoupledSide(sends=WallTemperature, receives=WallHeatFlux,
                    send=self.send, receive=self.receive, coordinates=points)
    """

    sends: type[BaseModel]
    receives: type[BaseModel]
    send: Callable[[], BaseModel]
    receive: Callable[[Any], None]
    coordinates: Any = None

    def __post_init__(self) -> None:
        for role in ("sends", "receives"):
            if not _is_model(getattr(self, role)):
                raise TypeError(
                    f"a coupled side declares what it {role} as a pydantic model, "
                    f"got {getattr(self, role)!r}"
                )

    @property
    def points(self) -> Optional[int]:
        """The number of points of the patch (rows of ``coordinates``), or ``None``."""
        if self.coordinates is None:
            return None
        return int(np.shape(self.coordinates)[0])


def _kinds(model: type[BaseModel]) -> dict[str, str]:
    """The kind of each field of the exchange model ``model``: ``Array``, or its type's name."""
    return {
        name: "Array"
        if any(isinstance(meta, _ArrayKind) for meta in info.metadata)
        else getattr(info.annotation, "__name__", repr(info.annotation))
        for name, info in model.model_fields.items()
    }


def _array_names(model: type[BaseModel]) -> tuple[str, ...]:
    """The :data:`Array` fields of the exchange model ``model``."""
    return tuple(name for name, kind in _kinds(model).items() if kind == "Array")


@dataclass(frozen=True)
class ExchangeSide:
    """A coupled side of a region: its region, patch and field, and what it exchanges.

    Built by :meth:`~neofoam.framework.coupling.Coupling.add` (read-only); ``key``
    is ``(region, patch, field)``.
    """

    region: str
    patch: str
    # the field the side couples, whose solution its region's coupling op repeats
    field: str
    side: CoupledSide

    @property
    def sends(self) -> type[BaseModel]:
        """The model of what the side sends."""
        return self.side.sends

    @property
    def receives(self) -> type[BaseModel]:
        """The model of what the side receives."""
        return self.side.receives

    @property
    def points(self) -> Optional[int]:
        """The number of points of the side's patch, or ``None`` without coordinates."""
        return self.side.points

    @property
    def mapped(self) -> tuple[str, ...]:
        """The fields of what the side receives that the mapping maps (its :data:`Array` fields).

        The others are one value for the interface, handed over as they are.
        """
        return _array_names(self.receives)

    @property
    def key(self) -> SideKey:
        """``(region, patch, field)``."""
        return (self.region, self.patch, self.field)

    @property
    def label(self) -> str:
        """``<region>.<patch>``."""
        return f"{self.region}.{self.patch}"


@dataclass(frozen=True)
class Exchange:
    """The exchange of a side, as its region sees it: its name, the partner and the mapping.

    The case configures it per patch pair (``system/regionCoupling.yaml``,
    ``exchanges:``). The side receives what the partner side, on patch
    ``partner[1]`` of region ``partner[0]``, sends, mapped onto its points by
    ``mapping``.
    """

    name: str
    partner: End
    mapping: Mapping


def _check_received(name: str, announced: dict[str, Any], side: ExchangeSide) -> None:
    """Raise ``ValueError`` unless ``side`` receives what its partner (``announced``) sends.

    By field name and kind (``Array`` or plain type); the model classes may differ.
    """
    sent, received = announced["kinds"], _kinds(side.receives)
    what_sent = f"{announced['model']} (sent by {announced['label']})"
    what_received = f"{side.receives.__qualname__} (received by {side.label})"
    problems = [f"'{n}' is in {what_sent} only" for n in sent if n not in received]
    problems += [f"'{n}' is in {what_received} only" for n in received if n not in sent]
    problems += [
        f"'{n}' is {kind} in {what_sent} but {received[n]} in {what_received}"
        for n, kind in sent.items()
        if n in received and received[n] != kind
    ]
    if problems:
        raise ValueError(
            f"exchange '{name}': each side must receive what the other sends: "
            + "; ".join(problems)
        )


def _arrays(names: Sequence[str], data: Data) -> Data:
    """The :data:`Array` fields ``names`` of ``data``."""
    return {name: data[name] for name in names}


def unchanged(changes: list[SideChange]) -> bool:
    """Converged once no side's data changes by more than round-off between two passes."""
    return all(
        np.allclose(change.current[name], change.previous[name], rtol=1e-12, atol=0.0)
        for change in changes
        for name in change.current
    )


@dataclass(frozen=True)
class CouplingOutcome:
    """How the coupling loop of one time step ended.

    ``passes`` is the number of passes it ran, the final one included,
    ``converged`` whether the convergence check held (``False``:
    ``max_iterations`` stopped it), and ``residuals`` the relative change of
    every field's exchanged :data:`Array` data in the last pass (largest over its
    sides and arrays; ``inf`` for a field no pass measured).
    """

    passes: int
    converged: bool
    residuals: dict[str, float] = field(default_factory=dict)


class CouplingNotConvergedWarning(RuntimeWarning):
    """A coupling loop reached ``max_iterations`` before its exchange converged."""


def relative_changes(changes: list[SideChange]) -> dict[str, float]:
    """The relative change of each field's data: ``|current - previous| / |current|``.

    Largest over the sides and arrays of the field (no floor: a field's value
    is its largest measured change; a NaN stays NaN); the absolute change where
    the current data is zero.
    """
    residuals: dict[str, float] = {}
    for change in changes:
        for name, current in change.current.items():
            current = np.asarray(current, dtype=float)
            delta = float(np.linalg.norm(current - np.asarray(change.previous[name], dtype=float)))
            norm = float(np.linalg.norm(current))
            value = delta / norm if norm > 0 else delta
            residuals[change.field] = float(np.maximum(residuals.get(change.field, value), value))
    return residuals


@dataclass(frozen=True)
class GroupSide:
    """A side of the leader's node, as its region told the leader at startup.

    ``arrays`` are the :data:`Array` fields of what it sends; ``partner`` is the
    index of its partner among the node's sides; ``exchange`` the name of its
    exchange.
    """

    region: str
    patch: str
    field: str
    arrays: tuple[str, ...]
    partner: int
    exchange: str = ""


class CouplingController:
    """The leader's part of a coupling node: decides for the node's regions, per pass.

    ``sides`` are the sides of the node, its regions in stepping order; ``label``
    names the leader's coupling op. After each pass (:meth:`decide`) it gets what
    every side sent and checks every measure of ``convergence`` on the change of
    its :data:`Array` data since the pass before (all must hold; none:
    :func:`unchanged`). Once they hold it runs exactly one more pass, the
    **final** one (:attr:`final`), and stops after it without checking again;
    the last pass ``max_iterations`` allows is final in advance (no pass more;
    it checks after it whether the data converged), and with ``max_iterations``
    1 the only pass is first and final. Before each pass, the final one too,
    it relaxes the :data:`Array` data carried into the next pass with the
    ``acceleration`` of the side's exchange (by exchange name; at most one per
    exchange, an exchange without one is not relaxed; plain fields pass through
    as they are).

    A side receives what its partner sent in this pass when the scheme is serial
    (the default) and the partner's region steps before its own; else
    (``parallel``, or the partner steps later) what the controller carried from
    the pass before. Only the carried data is relaxed.

    When a time step's coupling ends it appends its :class:`CouplingOutcome` to
    :attr:`outcomes` and hands it to the execution observer
    (``coupling_finished``); ending at ``max_iterations`` unconverged also warns
    (:class:`CouplingNotConvergedWarning`). Not thread safe.

    Raises ``ValueError`` when a measure of ``convergence`` on a field's data
    (``data``) has none to compare: no side of the field, or its sides send no
    :data:`Array` data; :meth:`decide` raises ``FloatingPointError`` when the
    change of a side is not finite.

    The leader builds it with its coupling's ``controller`` factory, which it
    hands the node (by default :func:`~neofoam.framework.coupling.node_controller`:
    the node's scheme, convergence, acceleration and max_iterations; one
    acceleration per exchange: it keeps the data of its last passes)::

        def controller(sides, label, *, node):
            return CouplingController(
                sides,
                label,
                max_iterations=50,
                parallel=node.parallel,
                acceleration={"interface": AitkenRelaxation(initial=0.5)},
            )
    """

    def __init__(
        self,
        sides: Sequence[GroupSide],
        label: str,
        convergence: Sequence[Convergence] = (),
        max_iterations: int = 100,
        parallel: bool = False,
        acceleration: Optional[dict[str, Acceleration]] = None,
    ) -> None:
        #: The name of the leader, the coupling op that owns the controller.
        self.label = label
        #: Whether a side always receives what its partner sent in the pass before.
        self.parallel = parallel
        self._sides = tuple(sides)
        self._convergence = tuple(convergence)
        self._max_iterations = max_iterations
        self._acceleration = dict(acceleration or {})
        self._check_convergence()
        rank: dict[str, int] = {}
        for side in self._sides:
            rank.setdefault(side.region, len(rank))
        self._fresh = [
            not parallel and rank[self._sides[side.partner].region] < rank[side.region]
            for side in self._sides
        ]
        self._outcomes: list[CouplingOutcome] = []
        # inf for every field with Array data until a pass measured its change:
        # nothing counts as converged before it was measured
        self._unmeasured = {side.field: math.inf for side in self._sides if side.arrays}
        self._residuals = dict(self._unmeasured)
        # the pass of the current time step; 0 between time steps
        self._iteration = 0
        # whether the pass started last is the final one, and whether it is final
        # because the data converged
        self._final = False
        self._converged_before_final = False
        # what each side carries into the pass (relaxed), in the order of _sides
        self._carried: list[Data] = []

    def _check_convergence(self) -> None:
        """Raise ``ValueError`` when a measure of the convergence has no data to compare.

        A measure on the data of a field (its ``data``, e.g. :class:`RelativeConvergence`)
        compares the :data:`Array` data the node's sides of that field send (plain
        fields are not compared); with none it would hold over nothing.
        """
        for measure in self._convergence:
            data = getattr(measure, "data", None)
            if not isinstance(data, str):
                continue
            of_field = [side for side in self._sides if side.field == data]
            if not of_field:
                raise ValueError(
                    f"coupling {self.label}: convergence {type(measure).__name__} on {data}, "
                    f"which no side of the node sends (its sides couple "
                    f"{sorted({side.field for side in self._sides})})"
                )
            if not any(side.arrays for side in of_field):
                raise ValueError(
                    f"coupling {self.label}: convergence {type(measure).__name__} on {data} "
                    f"has no Array data to compare: the sides of {data} send plain fields "
                    "only, which no convergence compares"
                )

    @property
    def sides(self) -> tuple[GroupSide, ...]:
        """The sides of the group, its regions in stepping order."""
        return self._sides

    @property
    def iteration(self) -> int:
        """The pass of the current time step's coupling; ``0`` between time steps."""
        return self._iteration

    @property
    def final(self) -> bool:
        """Whether the pass :meth:`decide` started last is the time step's final one."""
        return self._final

    @property
    def outcomes(self) -> tuple[CouplingOutcome, ...]:
        """The outcome of every finished coupling, one per time step."""
        return tuple(self._outcomes)

    @property
    def residuals(self) -> dict[str, float]:
        """The relative change of each field's :data:`Array` data in the last pass.

        ``inf`` for a field no pass of the time step's coupling measured yet.
        """
        return dict(self._residuals)

    def fresh(self, index: int) -> bool:
        """Whether side ``index`` receives what its partner sent in this pass, not carried data."""
        return self._fresh[index]

    def decide(self, current: list[Data]) -> Optional[list[Data]]:
        """The decision before a pass, from what every side sends now (in the order of ``sides``).

        Returns what each side carries into the pass (relaxed), :attr:`final`
        telling whether it is the final pass, or ``None``: the time step's
        coupling ended after its final pass, its outcome recorded.
        """
        if self._iteration == 0:
            self._carried = list(current)
            return self._start_pass(final=self._max_iterations <= 1)
        changes = [
            SideChange(side.field, _arrays(side.arrays, previous), _arrays(side.arrays, now))
            for side, previous, now in zip(self._sides, self._carried, current)
        ]
        self._residuals = {**self._unmeasured, **self._relative_changes(changes)}
        if self._final:
            converged = self._converged_before_final or self._converged(changes)
            self._finish(converged)
            return None
        self._carried = list(current)
        if self._converged(changes):
            self._converged_before_final = True
            return self._start_pass(final=True)
        return self._start_pass(final=self._iteration + 1 >= self._max_iterations)

    def _relative_changes(self, changes: list[SideChange]) -> dict[str, float]:
        """:func:`relative_changes` of ``changes`` (in the order of ``sides``).

        Raises ``FloatingPointError`` when the change of a side is not finite.
        """
        residuals: dict[str, float] = {}
        for side, change in zip(self._sides, changes):
            for name, value in relative_changes([change]).items():
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"coupling {self.label}: the relative change of the {name} of region "
                        f"{side.region} patch {side.patch} is not finite in pass {self._iteration}"
                    )
                residuals[name] = float(np.maximum(residuals.get(name, value), value))
        return residuals

    def _converged(self, changes: list[SideChange]) -> bool:
        """Whether every measure of the convergence holds (none: :func:`unchanged`)."""
        if not self._convergence:
            return bool(unchanged(changes))
        return all(bool(measure(changes)) for measure in self._convergence)

    def _finish(self, converged: bool) -> None:
        outcome = CouplingOutcome(self._iteration, converged, dict(self._residuals))
        self._outcomes.append(outcome)
        self._iteration = 0
        self._final = False
        self._converged_before_final = False
        observer = execution_observer.current()
        if observer is not None:
            observer.coupling_finished(self, outcome)
        if not converged:
            residuals = ", ".join(
                f"{name} {value:.3g}" for name, value in outcome.residuals.items()
            )
            warnings.warn(
                f"coupling {self.label} did not converge in {outcome.passes} passes "
                f"(max_iterations); relative change: {residuals or 'n/a'}",
                CouplingNotConvergedWarning,
                stacklevel=2,
            )

    def _start_pass(self, final: bool) -> list[Data]:
        if self._iteration == 0:
            self._residuals = dict(self._unmeasured)
        self._iteration += 1
        self._final = final
        for exchange, acceleration in self._acceleration.items():
            # the sides of the exchange whose partner receives their data as carried
            carried = [
                i
                for i, side in enumerate(self._sides)
                if side.exchange == exchange and not self._fresh[side.partner]
            ]
            if not carried:
                continue
            arrays = [_arrays(self._sides[i].arrays, self._carried[i]) for i in carried]
            relaxed = acceleration(arrays, self._iteration)
            for index, data in zip(carried, relaxed):
                self._carried[index] = {**self._carried[index], **data}
        return list(self._carried)


# The leader's decision for one region: another pass?, is it the final one?, and what
# each of its sides receives, by patch: the carried data, or None: what its partner
# sends in this pass.
_Decision = tuple[bool, bool, dict[str, Optional[Data]]]


class CouplingOp(IterativeOp):
    """A region's coupling node: a loop whose test exchanges the node's sides' data, per pass.

    Built from what the region knows of the case's coupling (``coupling``: its
    exchanges, the group's regions, its transport), the ``node`` (its exchanges)
    and the region's coupled ``sides`` of the node (the region's
    :class:`~neofoam.framework.coupling.Coupling` builds it); raises
    ``ValueError`` when a side has no exchange of the node, a patch two sides,
    or an exchange of the node in the region no side. The resolver places it
    as the node's placement for the region says (``metadata.wraps``). Its label
    is ``<region>.<node>``; its messages to other regions are tagged by node
    (``<node>/sides``, ``<node>/pass``, ``<node>/decision``) or by exchange
    (``side:<exchange>``, ``data:<exchange>``).

    Its loop test runs before each pass. The first test ever exchanges the
    sides' models and points with the partner regions and checks each exchange.
    Each test then sends what every side sends now: to the partner that steps
    later (serial: it receives it in the same pass) and to the node's leader;
    the leader decides for the node and tells every region to go on or stop, and
    whether the pass is the node's final one, which the test marks on the loop's
    entry (``Context.mark_final``: an operation marked ``final`` runs in it). On
    another pass every side gets what its partner sent (fresh, or carried by the
    leader), mapped onto its points with the exchange's mapping, validated as
    its ``receives`` model, reported to the execution observer
    (``coupling_exchanged``) and handed to its ``receive``. Not thread safe.
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
                link.name, link.partner(self.region), mapping(link.mapping)
            )
            for link in self._links
            if self.region in dict(link.ends)
        }
        self._sides = tuple(sides)
        self._check()
        self._rank = {region: index for index, region in enumerate(self._group)}
        # set at the first test: the partner's points per patch, and the leader's controller
        self._connected = False
        self._partner_points: dict[str, Any] = {}
        self._controller: Optional[CouplingController] = None
        # the leader: the patches of every region of the group, in the order of their sides
        self._patches: dict[str, list[str]] = {}
        self._iteration = 0

    def _check(self) -> None:
        """Raise ``ValueError`` unless every side has an exchange and every exchange one side."""
        patches = [side.patch for side in self._sides]
        for side in self._sides:
            if side.patch not in self._exchanges:
                raise ValueError(
                    f"no exchange of coupling node '{self.node.name}' connects the coupled side "
                    f"of {side.label}; a coupled boundary condition needs its exchange "
                    "(system/regionCoupling.yaml: exchanges)"
                )
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
        if not self._connected:
            self._connect()
        time = getattr(ctx.time, "value", None)
        current = [_sent(side, self.node.name, self._iteration, time) for side in self._sides]
        if self._iteration > 0 and not self.node.parallel:
            for side, data in zip(self._sides, current):
                region = self.exchange_of(side).partner[0]
                if self._rank[region] > self._rank[self.region]:
                    self._send(region, f"data:{self.exchange_of(side).name}", data)
        go_on, final, decided = self._decide(current, time if self._iteration == 0 else None)
        if not go_on:
            self._iteration = 0
            return False
        ctx.mark_final(final)
        self._iteration += 1
        for side in self._sides:
            self._hand_over(side, decided)
        return True

    def _send(self, target: str, tag: str, message: Any) -> None:
        self._transport.send(self.region, target, tag, message)

    def _receive(self, source: str, tag: str) -> Any:
        return self._transport.receive(source, self.region, tag)

    def _connect(self) -> None:
        """Tell each partner its side (models, points), the leader the region's sides; check.

        Raises ``ValueError`` when a side does not receive what its partner sends
        or its mapping cannot map between their points; in the leader, when the
        controller cannot be built.
        """
        for side in self._sides:
            exchange = self.exchange_of(side)
            announced = {
                "label": side.label,
                "model": side.sends.__qualname__,
                "kinds": _kinds(side.sends),
                "points": side.side.coordinates,
            }
            self._send(exchange.partner[0], f"side:{exchange.name}", announced)
        sides = [(side.patch, side.field, _array_names(side.sends)) for side in self._sides]
        if not self.is_leader:
            self._send(self._leader, self._tag("sides"), sides)
        for side in self._sides:
            exchange = self.exchange_of(side)
            announced = self._receive(exchange.partner[0], f"side:{exchange.name}")
            _check_received(exchange.name, announced, side)
            exchange.mapping.check(announced["points"], side.side.coordinates)
            self._partner_points[side.patch] = announced["points"]
        if self.is_leader:
            self._controller = self._coupling.controller(
                self._node_sides(sides),
                self.label,
                node=self.node,
            )
        self._connected = True

    def _tag(self, what: str) -> str:
        """The transport tag ``<node>/<what>`` of the node's messages."""
        return f"{self.node.name}/{what}"

    def _node_sides(self, own: list[tuple[str, str, tuple[str, ...]]]) -> list[GroupSide]:
        """The leader: the sides of every region of the node, as the regions told it."""
        told = {
            region: own if region == self.region else self._receive(region, self._tag("sides"))
            for region in self._group
        }
        ends = [(region, *side) for region, sides in told.items() for side in sides]
        self._patches = {
            region: [patch for patch, *_rest in sides] for region, sides in told.items()
        }
        index = {(region, patch): i for i, (region, patch, *_rest) in enumerate(ends)}
        other = {end: link.ends[1 - k] for link in self._links for k, end in enumerate(link.ends)}
        exchange = {end: link.name for link in self._links for end in link.ends}
        return [
            GroupSide(
                region,
                patch,
                field,
                arrays,
                index[other[(region, patch)]],
                exchange[(region, patch)],
            )
            for region, patch, field, arrays in ends
        ]

    def _decide(self, current: list[Data], time: Any) -> _Decision:
        """The node leader's decision for this region, from what its sides send now.

        ``time`` is the region's time as a time step's coupling starts, else ``None``.
        """
        leader = self._leader
        if not self.is_leader:
            self._send(leader, self._tag("pass"), (time, current))
            decision: _Decision = self._receive(leader, self._tag("decision"))
            return decision
        assert self._controller is not None
        sent = {
            region: (time, current)
            if region == self.region
            else self._receive(region, self._tag("pass"))
            for region in self._group
        }
        times = {region: when for region, (when, _data) in sent.items()}
        if len(set(times.values())) > 1:
            raise ValueError(
                f"the coupled regions {list(times)} start a time step at different times "
                f"{times}; they must step the same times (deltaT, endTime)"
            )
        carried = self._controller.decide(
            [data for _when, sides in sent.values() for data in sides]
        )
        decisions: dict[str, _Decision] = {}
        index = 0
        for region, patches in self._patches.items():
            received: dict[str, Optional[Data]] = {}
            for patch in patches:
                if carried is not None:
                    fresh = self._controller.fresh(index)
                    partner = self._controller.sides[index].partner
                    received[patch] = None if fresh else carried[partner]
                index += 1
            decisions[region] = (carried is not None, self._controller.final, received)
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
            data = self._receive(exchange.partner[0], f"data:{exchange.name}")
        mapped = exchange.mapping(
            _arrays(side.mapped, data), self._partner_points[side.patch], side.side.coordinates
        )
        received = side.receives.model_validate({**data, **mapped})
        observer = execution_observer.current()
        if observer is not None:
            observer.coupling_exchanged(self, side, received)
        side.side.receive(received)


def _sent(side: ExchangeSide, node: str, iteration: int, time: Any) -> Data:
    """What ``side`` sends now, as named copies.

    Raises ``FloatingPointError`` when an :data:`Array` of it is not finite (in
    coupling node ``node``, after pass ``iteration``, at ``time``).
    """
    sent = side.side.send()
    if not isinstance(sent, side.sends):
        raise TypeError(
            f"the coupled side {side.label} declares it sends {side.sends.__name__}, "
            f"but its send() returned {type(sent).__name__}"
        )
    data = {
        name: np.array(getattr(sent, name), dtype=float) if kind == "Array" else getattr(sent, name)
        for name, kind in _kinds(side.sends).items()
    }
    for name in _array_names(side.sends):
        if not np.all(np.isfinite(data[name])):
            when = f"after pass {iteration}" if iteration else "before the first pass"
            at = "" if time is None else f" (t={time})"
            raise FloatingPointError(
                f"coupling '{node}': the {name} of region {side.region} patch {side.patch} "
                f"is not finite {when}{at}"
            )
    return data
