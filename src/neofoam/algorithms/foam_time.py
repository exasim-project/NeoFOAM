# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pure-Python re-implementation of Foam::Time's advancement (the *stepper*).

The step update — physical time *or* iteration — is done entirely in Python so
every backend shares one implementation and behaves identically. Mirrors
``$FOAM_SRC/OpenFOAM/db/Time/{Time.C,TimeState.C}`` (v2406): ``operator++``,
``run``, ``loop``, ``end``, ``setDeltaT``/``adjustDeltaT`` — including the
old-time-step bookkeeping (``deltaT0``).

It does not write fields and does not delegate advancement to a backend. When a
backend (e.g. OpenFOAM) needs to know the new step — to keep its own ``Time`` in
sync so it writes into the right time directory — the stepper *pushes* that data
out through a :class:`StepSink`. Writing itself is a separate concern (the
``fieldWriter`` Model).
"""

from __future__ import annotations

import math
from typing import Optional, Protocol, runtime_checkable

from pydantic import Field, model_validator

from neofoam.algorithms.time_integration import (
    TimeIntegration,
    TransientIntegration,
)
from neofoam.algorithms.write_control import WriteControlConfig
from neofoam.io import OF, IOStrategy

SMALL = 1e-15
LABEL_MAX = 2**31 - 1


@IOStrategy(OF("system/controlDict"))
class TimeControlConfig(WriteControlConfig):
    """Time-stepping + write control, as a validated config.

    Extends :class:`~neofoam.algorithms.write_control.WriteControlConfig` with the
    advancement keys the stepper and loop read (``endTime``/``deltaT`` and the
    adaptive-stepping cap). The concrete, file-bound config subclasses this and
    adds an ``@IOStrategy``; the framework only ever sees the validated config,
    never a raw dict — so a plugin extends the loop by extending this schema.
    """

    endTime: float = Field(gt=0)
    deltaT: float = Field(gt=0)
    adjustTimeStep: bool = False
    maxCo: Optional[float] = Field(default=None, gt=0)
    maxDeltaT: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_adjust_time_step(self) -> "TimeControlConfig":
        if self.adjustTimeStep and self.maxCo is None:
            raise ValueError("maxCo must be set when adjustTimeStep=True")
        return self


@runtime_checkable
class StepSink(Protocol):
    """One-way seam: the stepper *pushes* each step update to other functionality.

    The Python stepper owns the advancement; a backend that must mirror it
    (OpenFOAM keeping ``Foam::Time`` in sync so field IO lands in the right time
    directory) implements this sink. It is push-only and does **not** write
    fields — writing is the ``fieldWriter`` Model's job.
    """

    def set_delta_t(self, dt: float) -> None: ...
    def advance_to(self, value: float, index: int) -> None: ...


class NullStepSink:
    """Default sink for standalone / pure-Python use: discards the updates."""

    def set_delta_t(self, dt: float) -> None:
        return None

    def advance_to(self, value: float, index: int) -> None:
        return None


# write-control kinds (canonical OpenFOAM keywords + the "adjustable" alias)
_TIME_STEP = "timeStep"
_RUN_TIME = "runTime"
_ADJUSTABLE = "adjustableRunTime"
_NONE = "none"
_WRITE_CONTROL_ALIASES = {
    "timeStep": _TIME_STEP,
    "runTime": _RUN_TIME,
    "adjustable": _ADJUSTABLE,
    "adjustableRunTime": _ADJUSTABLE,
    "none": _NONE,
}


def _round_half_away(x: float) -> int:
    """Foam::round — round half away from zero (≠ Python banker's round)."""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


class FoamTime:
    """The pure-Python stepper — advances time/iteration, pushes to a StepSink."""

    def __init__(
        self,
        *,
        start_time: float = 0.0,
        end_time: float,
        delta_t: float,
        write_control: str = "timeStep",
        write_interval: float = 1.0,
        max_delta_t: float = 1e30,
        precision: int = 6,
        integration: Optional[TimeIntegration] = None,
        sink: Optional[StepSink] = None,
    ) -> None:
        # the integration owns the regime-specific decisions (initial step,
        # step naming); transient is the default for a bare-deltaT stepper.
        self._integration: TimeIntegration = (
            integration if integration is not None else TransientIntegration()
        )
        self._start = start_time
        self._end = end_time
        self._delta_t = self._integration.initial_delta_t(delta_t)
        self._write_control = _WRITE_CONTROL_ALIASES.get(write_control, write_control)
        self._write_interval = write_interval
        self._max_delta_t = max_delta_t
        self._precision = precision
        self._sink: StepSink = sink if sink is not None else NullStepSink()

        self._value = start_time
        self._index = 0
        self._write_time_index = 0
        self._delta_t0 = 0.0
        self._delta_t_save = 0.0
        self._delta_t_changed = False
        self._write_time = False

    # -- construction from a validated config -----------------------------
    @classmethod
    def from_config(
        cls,
        config: TimeControlConfig,
        integration: Optional[TimeIntegration] = None,
        sink: Optional[StepSink] = None,
    ) -> "FoamTime":
        return cls(
            start_time=config.startTime,
            end_time=config.endTime,
            delta_t=config.deltaT,
            write_control=config.writeControl,
            write_interval=config.writeInterval,
            max_delta_t=config.maxDeltaT if config.maxDeltaT is not None else 1e30,
            integration=integration,
            sink=sink,
        )

    # -- backend seam -----------------------------------------------------
    def set_sink(self, sink: StepSink) -> None:
        """Inject the backend sink after construction (e.g. a solver wiring its
        backend ``Time`` to mirror the pure-Python advancement). Standalone use
        keeps the default :class:`NullStepSink`."""
        self._sink = sink

    # -- queries ----------------------------------------------------------
    def value(self) -> float:
        return self._value

    def deltaTValue(self) -> float:
        return self._delta_t

    def deltaT0Value(self) -> float:
        return self._delta_t0

    def timeIndex(self) -> int:
        return self._index

    def outputTime(self) -> bool:
        return self._write_time

    def run(self) -> bool:
        return self._value < (self._end - 0.5 * self._delta_t)

    def end(self) -> bool:
        return self._value > (self._end + 0.5 * self._delta_t)

    def timeName(self) -> str:
        return self._integration.step_name(self._value, self._index, self._precision)

    # -- mutators ---------------------------------------------------------
    def setDeltaT(self, dt: float, adjust: bool = True) -> None:
        self._delta_t = dt
        self._delta_t_changed = True
        if adjust:
            self._adjust_delta_t()
        # push the new step size to the backend (no-op for standalone)
        self._sink.set_delta_t(self._delta_t)

    def _adjust_delta_t(self) -> None:
        if self._write_control != _ADJUSTABLE:
            return
        time_to_next_write = max(
            0.0,
            (self._write_time_index + 1) * self._write_interval
            - (self._value - self._start),
        )
        n_steps = time_to_next_write / self._delta_t
        if n_steps < LABEL_MAX:
            n_steps_to_next = max(1, _round_half_away(n_steps))
            new_delta_t = time_to_next_write / n_steps_to_next
            if new_delta_t >= self._delta_t:
                self._delta_t = min(new_delta_t, 2.0 * self._delta_t)
            else:
                self._delta_t = max(new_delta_t, 0.2 * self._delta_t)

    def increment(self) -> None:
        """``Foam::Time::operator++`` — advance, roll the old time, set writeTime."""
        self._delta_t0 = self._delta_t_save
        self._delta_t_save = self._delta_t

        self._value = self._value + self._delta_t
        self._index += 1

        if abs(self._value) < 10 * SMALL * self._delta_t:
            self._value = 0.0

        self._write_time = False
        if self._write_control == _TIME_STEP:
            self._write_time = (self._index % int(self._write_interval)) == 0
        elif self._write_control in (_RUN_TIME, _ADJUSTABLE):
            write_index = int(
                ((self._value - self._start) + 0.5 * self._delta_t)
                / self._write_interval
            )
            if write_index > self._write_time_index:
                self._write_time = True
                self._write_time_index = write_index

        # push the advanced step to the backend so it can mirror our time
        self._sink.advance_to(self._value, self._index)

    def loop(self) -> bool:
        running = self.run()
        if running:
            self.increment()
        return running

    def stop(self) -> None:
        """End the run now — advancement only, no write.

        Whether the final step is written is the ``WriteControl``'s decision
        (single responsibility); ending the run must not force a write.
        """
        self._end = self._value

    def __call__(self, delta_t: Optional[float] = None) -> None:
        """``time()`` advances one step; ``time(dt)`` sets the step then advances."""
        if delta_t is not None:
            self.setDeltaT(delta_t)
        self.increment()
