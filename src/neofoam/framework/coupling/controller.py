# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The leader's part of a coupling node: its :class:`CouplingController`.

The **leader** of a node, the last region in stepping order of the regions the
node's exchanges connect, owns the node's controller, which decides after each
pass whether the node's data converged, and relaxes the data carried into the
next pass, per exchange. It also decides which pass is the node's **final**
one: the pass after the one its data converged in, or the last one
``max_iterations`` allows. It knows every side of the node by its end
(:data:`~neofoam.framework.coupling.data.End`, ``(region, patch)``), as the
side's region told the leader (:class:`GroupSide`).
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from neofoam.framework import execution_observer

from .data import End, _arrays
from .kernels import CouplingAcceleration, CouplingConvergence, SideChange
from .transport import Data


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
    """A coupling loop reached ``max_iterations`` before its exchange converged.

    It points at the code that ran the coupled solver: the first caller outside
    the framework (:func:`_stacklevel`), e.g. the solver's ``run``.
    """


# The framework's directory: the frames a coupling warning passes over.
_FRAMEWORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _stacklevel() -> int:
    """The ``stacklevel`` of ``warnings.warn``, called by this function's caller, that
    points at the first frame outside the framework (``neofoam.framework``)."""
    level, frame = 1, sys._getframe(1)
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_FRAMEWORK):
        level, frame = level + 1, frame.f_back
    return level


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
    end ``(region, patch)`` of its partner side; ``exchange`` the name of its
    exchange.
    """

    region: str
    patch: str
    field: str
    arrays: tuple[str, ...]
    partner: End
    exchange: str = ""

    @property
    def end(self) -> End:
        """``(region, patch)``: the side's end of its exchange."""
        return (self.region, self.patch)


class CouplingController:
    """The leader's part of a coupling node: decides for the node's regions, per pass.

    ``sides`` are the sides of the node, its regions in stepping order; ``label``
    names the leader's coupling op. After each pass (:meth:`decide`) it gets what
    every side sent, and the number of passes the leader's op ran in the time
    step (its entry on ``Context.loops``, the one count of passes), and checks
    every measure of ``convergence`` on the change of its :data:`Array` data
    since the pass before (all must hold; none:
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
    the pass before. Only the carried data is relaxed. This is the only rule: the
    sending regions send their data fresh exactly where the leader decided it.

    When a time step's coupling ends it appends its :class:`CouplingOutcome` to
    :attr:`outcomes` and hands it to the execution observer
    (``coupling_finished``); ending at ``max_iterations`` unconverged also warns
    (:class:`CouplingNotConvergedWarning`). Not thread safe.

    Raises ``ValueError`` when a measure of ``convergence`` on a field's data
    (``data``) has none to compare: no side of the field, or its sides send no
    :data:`Array` data; :meth:`decide` raises ``FloatingPointError`` when the
    change of a side is not finite.

    The leader builds it with its coupling's ``controller`` factory, which it
    hands the node (by default :func:`~neofoam.framework.coupling.region.node_controller`:
    the node's scheme, convergence, acceleration and max_iterations; one
    acceleration per exchange: it keeps the data of its last passes)::

        def controller(sides, label, *, node):
            return CouplingController(
                sides,
                label,
                max_iterations=50,
                parallel=node.parallel,
                acceleration={"interface": AitkenRelaxation(relaxation=0.5)},
            )
    """

    def __init__(
        self,
        sides: Sequence[GroupSide],
        label: str,
        convergence: Sequence[CouplingConvergence] = (),
        max_iterations: int = 100,
        parallel: bool = False,
        acceleration: Optional[dict[str, CouplingAcceleration]] = None,
    ) -> None:
        #: The name of the leader, the coupling op that owns the controller.
        self.label = label
        #: Whether a side always receives what its partner sent in the pass before.
        self.parallel = parallel
        self._sides = {side.end: side for side in sides}
        self._convergence = tuple(convergence)
        self._max_iterations = max_iterations
        self._acceleration = dict(acceleration or {})
        self._check_convergence()
        rank: dict[str, int] = {}
        for side in self._sides.values():
            rank.setdefault(side.region, len(rank))
        self._fresh = {
            end: not parallel and rank[side.partner[0]] < rank[side.region]
            for end, side in self._sides.items()
        }
        self._outcomes: list[CouplingOutcome] = []
        # inf for every field with Array data until a pass measured its change:
        # nothing counts as converged before it was measured
        self._unmeasured = {side.field: math.inf for side in self._sides.values() if side.arrays}
        self._residuals = dict(self._unmeasured)
        # whether the pass started last is the final one, and whether it is final
        # because the data converged
        self._final = False
        self._converged_before_final = False
        # what each side carries into the pass (relaxed), by its end
        self._carried: dict[End, Data] = {}

    def _check_convergence(self) -> None:
        """Raise ``ValueError`` when a measure of the convergence has no data to compare.

        A measure on the data of a field (its ``data``, e.g. :class:`RelativeConvergence`)
        compares the :data:`Array` data the node's sides of that field send (plain
        fields are not compared); with none it would hold over nothing.
        """
        sides = self._sides.values()
        for measure in self._convergence:
            data = getattr(measure, "data", None)
            if not isinstance(data, str):
                continue
            of_field = [side for side in sides if side.field == data]
            if not of_field:
                raise ValueError(
                    f"coupling {self.label}: convergence {type(measure).__name__} on {data}, "
                    f"which no side of the node sends (its sides couple "
                    f"{sorted({side.field for side in sides})})"
                )
            if not any(side.arrays for side in of_field):
                raise ValueError(
                    f"coupling {self.label}: convergence {type(measure).__name__} on {data} "
                    f"has no Array data to compare: the sides of {data} send plain fields "
                    "only, which no convergence compares"
                )

    @property
    def sides(self) -> dict[End, GroupSide]:
        """The sides of the group by end, its regions in stepping order."""
        return dict(self._sides)

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

    def fresh(self, end: End) -> bool:
        """Whether the side at ``end`` receives what its partner sent in this pass, not carried."""
        return self._fresh[end]

    def decide(self, current: dict[End, Data], iteration: int) -> Optional[dict[End, Data]]:
        """The decision before a pass, from what every side sends now (by its end).

        ``iteration`` is the number of passes the time step's coupling ran
        (``LoopEntry.iteration`` of the leader's op): ``0`` before its first.
        Returns what each side carries into the pass (relaxed), by its end,
        :attr:`final` telling whether it is the final pass, or ``None``: the
        time step's coupling ended after its final pass, its outcome recorded.
        """
        current = {end: current[end] for end in self._sides}
        if iteration == 0:
            self._carried = current
            return self._start_pass(iteration + 1, final=self._max_iterations <= 1)
        changes = {
            end: SideChange(
                side.field,
                _arrays(side.arrays, self._carried[end]),
                _arrays(side.arrays, current[end]),
            )
            for end, side in self._sides.items()
        }
        self._residuals = {**self._unmeasured, **self._relative_changes(changes, iteration)}
        if self._final:
            converged = self._converged_before_final or self._converged(changes)
            self._finish(iteration, converged)
            return None
        self._carried = current
        if self._converged(changes):
            self._converged_before_final = True
            return self._start_pass(iteration + 1, final=True)
        return self._start_pass(iteration + 1, final=iteration + 1 >= self._max_iterations)

    def _relative_changes(self, changes: dict[End, SideChange], iteration: int) -> dict[str, float]:
        """:func:`relative_changes` of ``changes`` (by the end of each side).

        Raises ``FloatingPointError`` when the change of a side is not finite.
        """
        residuals: dict[str, float] = {}
        for (region, patch), change in changes.items():
            for name, value in relative_changes([change]).items():
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"coupling {self.label}: the relative change of the {name} of region "
                        f"{region} patch {patch} is not finite in pass {iteration}"
                    )
                residuals[name] = float(np.maximum(residuals.get(name, value), value))
        return residuals

    def _converged(self, changes: dict[End, SideChange]) -> bool:
        """Whether every measure of the convergence holds (none: :func:`unchanged`)."""
        listed = list(changes.values())
        if not self._convergence:
            return bool(unchanged(listed))
        return all(bool(measure(listed)) for measure in self._convergence)

    def _finish(self, passes: int, converged: bool) -> None:
        outcome = CouplingOutcome(passes, converged, dict(self._residuals))
        self._outcomes.append(outcome)
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
                stacklevel=_stacklevel(),
            )

    def _start_pass(self, number: int, final: bool) -> dict[End, Data]:
        """Start pass ``number`` (from 1) of the time step's coupling: what each side carries."""
        if number == 1:
            self._residuals = dict(self._unmeasured)
        self._final = final
        for exchange, acceleration in self._acceleration.items():
            # the sides of the exchange whose partner receives their data as carried
            carried = [
                end
                for end, side in self._sides.items()
                if side.exchange == exchange and not self._fresh[side.partner]
            ]
            if not carried:
                continue
            arrays = [_arrays(self._sides[end].arrays, self._carried[end]) for end in carried]
            relaxed = acceleration(arrays, number)
            for end, data in zip(carried, relaxed):
                self._carried[end] = {**self._carried[end], **data}
        return dict(self._carried)
