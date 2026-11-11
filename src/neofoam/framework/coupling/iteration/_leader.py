# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The leader's part of a coupling node: its :class:`Leader`."""

from __future__ import annotations

import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Optional, Sequence

import numpy as np

from neofoam.framework import execution_observer

from ..comm.channel import Data
from ..data import End
from ..exchanges import CouplingLoop
from ..strategies.convergence import SideChange


def unchanged(changes: list[SideChange]) -> bool:
    """Converged once no stream's data changes by more than round-off between two passes."""
    return all(
        np.allclose(change.current[name], change.previous[name], rtol=1e-12, atol=0.0)
        for change in changes
        for name in change.current
    )


@dataclass(frozen=True)
class CouplingOutcome:
    """How the coupling loop of one time step ended.

    ``passes`` includes the final one; ``residuals`` is each data field's relative
    change in the last pass (``inf`` where none measured it).
    """

    passes: int
    converged: bool
    residuals: dict[str, float] = field(default_factory=dict)


class CouplingNotConvergedWarning(RuntimeWarning):
    """A coupling loop reached ``max_iterations`` before it converged."""


_FRAMEWORK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep


def _stacklevel() -> int:
    """The ``warnings.warn`` stacklevel of the first frame outside ``neofoam.framework``."""
    level = 1
    frame: Optional[FrameType] = sys._getframe(1)
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_FRAMEWORK):
        level, frame = level + 1, frame.f_back
    return level


def relative_changes(changes: list[SideChange]) -> dict[str, float]:
    """Each data field's largest ``|current - previous| / |current|`` over its streams.

    The absolute change where the current data is zero; a NaN stays NaN.
    """
    residuals: dict[str, float] = {}
    for change in changes:
        for name, current in change.current.items():
            current = np.asarray(current, dtype=float)
            delta = float(np.linalg.norm(current - np.asarray(change.previous[name], dtype=float)))
            norm = float(np.linalg.norm(current))
            value = delta / norm if norm > 0 else delta
            residuals[name] = float(np.maximum(residuals.get(name, value), value))
    return residuals


# (region, port, data key) of what a port sends
StreamId = tuple[str, str, str]

# the loops enclosing a coupling op, outermost first: (name, iteration from 0)
Loops = tuple[tuple[Optional[str], int], ...]

# a stream of this region: (port, data key)
LocalStream = tuple[str, str]


# the leader's decision for a region: another pass?, final?, by received stream the
# carried data or None (receive it fresh), and the sent streams to send fresh
Decision = tuple[bool, bool, dict[LocalStream, Optional[Data]], frozenset[LocalStream]]


def _state(loops: Loops) -> str:
    """``time_loop#3 > ab#1`` (passes from 1), or ``top``."""
    return " > ".join(f"{name}#{iteration + 1}" for name, iteration in loops) or "top"


@dataclass(frozen=True)
class GroupStream:
    """What a port of the leader's node sends, as its region told the leader at startup."""

    region: str
    port: str
    key: str
    names: tuple[str, ...]
    partner: End
    exchange: str

    @property
    def id(self) -> StreamId:
        """``(region, port, key)``."""
        return (self.region, self.port, self.key)


class Leader:
    """The leader's part of a coupling node: after each pass, whether another runs and which.

    Final pass as :class:`CouplingLoop` describes; warns when more than one of
    ``max_iterations`` ends the loop unconverged (one is explicit coupling). A port
    receives a stream fresh when the node is serial and its sender steps earlier,
    else the carried data, relaxed. Not thread safe.
    """

    def __init__(
        self,
        streams: Sequence[GroupStream],
        label: str,
        node: CouplingLoop,
        group: Sequence[str],
    ) -> None:
        self.label = label
        self._name = node.name
        self._group = tuple(group)
        parallel = node.parallel
        self._streams = {stream.id: stream for stream in streams}
        self._convergence = tuple(node.convergence)
        self._max_iterations = node.max_iterations
        self._acceleration = dict(node.acceleration)
        rank = {region: index for index, region in enumerate(group)}
        # whether a stream's receiver takes what it sent in this pass
        self._fresh = {
            stream_id: not parallel and rank[stream.region] < rank[stream.partner[0]]
            for stream_id, stream in self._streams.items()
        }
        self._outcome: Optional[CouplingOutcome] = None
        # nothing counts as converged before a pass measured it
        self._unmeasured = {
            name: math.inf for stream in self._streams.values() for name in stream.names
        }
        self._check_convergence()
        self._residuals = dict(self._unmeasured)
        self._final = False
        self._converged_before_final = False
        self._carried: dict[StreamId, Data] = {}

    def _check_convergence(self) -> None:
        names = set(self._unmeasured)
        for measure in self._convergence:
            data = measure.data
            if data is not None and data not in names:
                raise ValueError(
                    f"coupling {self.label}: convergence {type(measure).__name__} on {data}, "
                    f"which no port of the node sends (they send {sorted(names)})"
                )

    @property
    def streams(self) -> dict[StreamId, GroupStream]:
        """What the node's ports send, by stream, in stepping order."""
        return dict(self._streams)

    @property
    def final(self) -> bool:
        """Whether the pass :meth:`decide` started last is the final one."""
        return self._final

    @property
    def outcome(self) -> Optional[CouplingOutcome]:
        """The outcome of the last finished coupling (time step); ``None`` before.

        Only the last one is kept; an execution observer gets every one (``coupling_finished``).
        """
        return self._outcome

    @property
    def residuals(self) -> dict[str, float]:
        """Each field's relative change in the last pass; ``inf`` before one measured it."""
        return dict(self._residuals)

    def fresh(self, stream: StreamId) -> bool:
        """Whether the receiver of ``stream`` takes what it sent in this pass."""
        return self._fresh[stream]

    def decisions(self, sent_by: dict[str, Any], iteration: int) -> dict[str, Decision]:
        """Each region's decision on the pass the regions sent: what it receives, sends fresh."""
        self._check_pass(sent_by, iteration)
        current: dict[StreamId, Data] = {
            (region, port, key): data
            for region, (_when, _loops, streams) in sent_by.items()
            for (port, key), data in streams.items()
        }
        carried = self.decide(current, iteration)
        received: dict[str, dict[LocalStream, Optional[Data]]] = {
            region: {} for region in self._group
        }
        fresh: dict[str, set[LocalStream]] = {region: set() for region in self._group}
        if carried is not None:
            for stream_id, stream in self.streams.items():
                partner_region, partner_port = stream.partner
                into = received[partner_region]
                if self.fresh(stream_id):
                    into[(partner_port, stream.key)] = None
                    fresh[stream.region].add((stream.port, stream.key))
                else:
                    into[(partner_port, stream.key)] = carried[stream_id]
        return {
            region: (carried is not None, self.final, by_stream, frozenset(fresh[region]))
            for region, by_stream in received.items()
        }

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
                f"coupling '{self._name}': its regions entered pass {iteration + 1} "
                f"in different loop states: {where}"
            )

    def decide(
        self, current: dict[StreamId, Data], iteration: int
    ) -> Optional[dict[StreamId, Data]]:
        """Each stream's carried data for the next pass; ``None`` at the end."""
        current = {stream_id: current[stream_id] for stream_id in self._streams}
        if iteration == 0:
            self._carried = current
            return self._start_pass(iteration + 1, final=self._max_iterations <= 1)
        changes = {
            stream_id: SideChange(self._carried[stream_id], current[stream_id])
            for stream_id in self._streams
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

    def _relative_changes(
        self, changes: dict[StreamId, SideChange], iteration: int
    ) -> dict[str, float]:
        """:func:`relative_changes`; raises ``FloatingPointError`` for a non-finite one."""
        residuals: dict[str, float] = {}
        for (region, port, _key), change in changes.items():
            for name, value in relative_changes([change]).items():
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"coupling {self.label}: the relative change of the {name} of "
                        f"{region}.{port} is not finite in pass {iteration}"
                    )
                residuals[name] = float(np.maximum(residuals.get(name, value), value))
        return residuals

    def _converged(self, changes: dict[StreamId, SideChange]) -> bool:
        """Whether every measure of the convergence holds (none: :func:`unchanged`)."""
        listed = list(changes.values())
        if not self._convergence:
            return bool(unchanged(listed))
        return all(bool(measure(listed)) for measure in self._convergence)

    def _finish(self, passes: int, converged: bool) -> None:
        outcome = CouplingOutcome(passes, converged, dict(self._residuals))
        self._outcome = outcome
        self._final = False
        self._converged_before_final = False
        observer = execution_observer.current()
        if observer is not None:
            observer.coupling_finished(self, outcome)
        if not converged and self._max_iterations > 1:
            residuals = ", ".join(
                f"{name} {value:.3g}" for name, value in outcome.residuals.items()
            )
            warnings.warn(
                f"coupling {self.label} did not converge in {outcome.passes} passes "
                f"(max_iterations); relative change: {residuals or 'n/a'}",
                CouplingNotConvergedWarning,
                stacklevel=_stacklevel(),
            )

    def _start_pass(self, number: int, final: bool) -> dict[StreamId, Data]:
        """Start pass ``number`` (from 1): what each stream carries, relaxed."""
        if number == 1:
            self._residuals = dict(self._unmeasured)
        self._final = final
        for exchange_name, acceleration in self._acceleration.items():
            carried = [
                stream_id
                for stream_id, stream in self._streams.items()
                if stream.exchange == exchange_name and not self._fresh[stream_id]
            ]
            if not carried:
                continue
            relaxed = acceleration([self._carried[stream_id] for stream_id in carried], number)
            self._carried.update(zip(carried, relaxed))
        return dict(self._carried)
