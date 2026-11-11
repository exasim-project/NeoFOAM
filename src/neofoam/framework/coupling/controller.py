# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The leader's part of a coupling node: its :class:`CouplingController`."""

from __future__ import annotations

import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from types import FrameType
from typing import Optional, Sequence

import numpy as np

from neofoam.framework import execution_observer

from .data import End
from .kernels import SideChange
from .region import CouplingNode
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

    ``passes`` includes the final one; ``converged`` is ``False`` when
    ``max_iterations`` stopped it; ``residuals`` is each field's relative change
    in the last pass (``inf`` where no pass measured it).
    """

    passes: int
    converged: bool
    residuals: dict[str, float] = field(default_factory=dict)


class CouplingNotConvergedWarning(RuntimeWarning):
    """A coupling loop reached ``max_iterations`` before it converged."""


# the frames a coupling warning passes over
_FRAMEWORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _stacklevel() -> int:
    """The ``warnings.warn`` stacklevel of the first frame outside ``neofoam.framework``."""
    level = 1
    frame: Optional[FrameType] = sys._getframe(1)
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_FRAMEWORK):
        level, frame = level + 1, frame.f_back
    return level


def relative_changes(changes: list[SideChange]) -> dict[str, float]:
    """Each field's largest ``|current - previous| / |current|`` over its sides and arrays.

    The absolute change where the current data is zero; a NaN stays NaN.
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
    """A side of the leader's node, as its region told the leader at startup."""

    region: str
    patch: str
    field: str
    partner: End
    exchange: str

    @property
    def end(self) -> End:
        """``(region, patch)``: the side's end of its exchange."""
        return (self.region, self.patch)


class CouplingController:
    """The leader's part of a coupling node: after each pass, whether another runs and which.

    :meth:`decide` gets what every side sent; once every convergence measure of
    the node holds, exactly one more pass runs, the final one, then it stops;
    the last pass ``max_iterations`` allows is final in advance. It relaxes the
    carried data with each exchange's acceleration. A side receives fresh data
    (sent in this pass) when the node is serial and its partner steps earlier,
    else what the controller carried; the sending regions follow this decision.
    A finished time step's :class:`CouplingOutcome` goes to :attr:`outcomes` and
    the execution observer, and warns when unconverged. Not thread safe.

    Raises ``ValueError`` for a convergence on a field no side sends;
    :meth:`decide` raises ``FloatingPointError`` for a non-finite change.
    """

    def __init__(self, sides: Sequence[GroupSide], label: str, node: CouplingNode) -> None:
        #: the label of the leader's coupling op
        self.label = label
        self.parallel = parallel = node.parallel
        self._sides = {side.end: side for side in sides}
        self._convergence = tuple(node.convergence)
        self._max_iterations = node.max_iterations
        self._acceleration = dict(node.acceleration)
        self._check_convergence()
        rank: dict[str, int] = {}
        for side in self._sides.values():
            rank.setdefault(side.region, len(rank))
        self._fresh = {
            end: not parallel and rank[side.partner[0]] < rank[side.region]
            for end, side in self._sides.items()
        }
        self._outcomes: list[CouplingOutcome] = []
        # nothing counts as converged before a pass measured it
        self._unmeasured = {side.field: math.inf for side in self._sides.values()}
        self._residuals = dict(self._unmeasured)
        self._final = False
        self._converged_before_final = False
        self._carried: dict[End, Data] = {}

    def _check_convergence(self) -> None:
        """Raise ``ValueError`` when a measure on a field's ``data`` has no side of it."""
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

    @property
    def sides(self) -> dict[End, GroupSide]:
        """The node's sides by end, in stepping order."""
        return dict(self._sides)

    @property
    def final(self) -> bool:
        """Whether the pass :meth:`decide` started last is the final one."""
        return self._final

    @property
    def outcomes(self) -> tuple[CouplingOutcome, ...]:
        """The outcome of every finished coupling, one per time step."""
        return tuple(self._outcomes)

    @property
    def residuals(self) -> dict[str, float]:
        """Each field's relative change in the last pass; ``inf`` before one measured it."""
        return dict(self._residuals)

    def fresh(self, end: End) -> bool:
        """Whether the side at ``end`` receives what its partner sent in this pass."""
        return self._fresh[end]

    def decide(self, current: dict[End, Data], iteration: int) -> Optional[dict[End, Data]]:
        """What each side carries into the next pass, by end, or ``None`` once it ended.

        ``current`` is what every side sends now, ``iteration`` the passes run so far.
        """
        current = {end: current[end] for end in self._sides}
        if iteration == 0:
            self._carried = current
            return self._start_pass(iteration + 1, final=self._max_iterations <= 1)
        changes = {
            end: SideChange(side.field, self._carried[end], current[end])
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
        """:func:`relative_changes`; raises ``FloatingPointError`` for a non-finite one."""
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
        """Start pass ``number`` (from 1): what each side carries, relaxed."""
        if number == 1:
            self._residuals = dict(self._unmeasured)
        self._final = final
        for exchange, acceleration in self._acceleration.items():
            carried = [
                end
                for end, side in self._sides.items()
                if side.exchange == exchange and not self._fresh[side.partner]
            ]
            if not carried:
                continue
            relaxed = acceleration([self._carried[end] for end in carried], number)
            self._carried.update(zip(carried, relaxed))
        return dict(self._carried)
