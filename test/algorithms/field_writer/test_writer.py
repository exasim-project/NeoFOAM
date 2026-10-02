# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the FieldWriter — persisting fields, the symmetric twin of the loop.

``Writer`` is the abstraction (implement it to add new writers); ``FieldWriter``
is the concrete one for simulation fields. It owns a WriteControl *policy* (when
to write, read off the ``LoopState``) and writes through a :class:`FieldHook`
*backend* (the action); the action runs only on write steps.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import Field

from neofoam.algorithms.field_writer.write_control import (
    IntervalWriteControl,
    StepperWriteControl,
)
from neofoam.algorithms.field_writer.writer import FieldHook, FieldWriter, Writer
from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.algorithms.solution_loop.solution_loop import SolutionLoop


class FakeFieldHook(FieldHook):
    """Records the fields handed to it on each write (a fake backend)."""

    calls: list[list[str]] = Field(default_factory=list)

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        self.calls.append(sorted(fields))


_FIELDS: dict[str, Any] = {"U": object(), "p": object()}


def _loop(
    *, end_time: float = 0.4, delta_t: float = 0.1, write_interval: float = 1.0
) -> SolutionLoop:
    """A SolutionLoop over a fresh LoopState; ``loop.state`` is the StepView."""
    return SolutionLoop(
        state=LoopState(
            value=0.0,
            delta_t=delta_t,
            end_time=end_time,
            write_control="timeStep",
            write_interval=write_interval,
        )
    )


def test_field_writer_is_a_writer() -> None:
    assert isinstance(FieldWriter(write_control=StepperWriteControl()), Writer)


def test_writes_fields_on_write_steps_only() -> None:
    hook = FakeFieldHook()
    writer = FieldWriter(write_control=IntervalWriteControl(interval=2), hook=hook)
    loop = _loop()
    while loop.run():
        loop.advance()
        writer.write(loop.state, _FIELDS)
    # indices 1,2,3,4 -> write on 2 and 4, each time handed all the fields
    assert hook.calls == [["U", "p"], ["U", "p"]]


def test_write_returns_whether_it_wrote() -> None:
    hook = FakeFieldHook()
    writer = FieldWriter(write_control=StepperWriteControl(), hook=hook)
    loop = _loop(end_time=0.2, write_interval=1)
    loop.advance()  # index 1, a write step (writeInterval 1)
    assert writer.write(loop.state, _FIELDS) is True
    assert hook.calls == [["U", "p"]]


def test_should_write_does_not_write() -> None:
    hook = FakeFieldHook()
    writer = FieldWriter(write_control=IntervalWriteControl(interval=2), hook=hook)
    loop = _loop()
    loop.advance()  # index 1 -> not a write step for interval 2
    assert writer.should_write(loop.state) is False
    assert hook.calls == []


def test_null_hook_is_the_default() -> None:
    # no hook -> decision still works, nothing is persisted (standalone use)
    writer = FieldWriter(write_control=StepperWriteControl())
    loop = _loop(write_interval=1)
    loop.advance()
    assert writer.write(loop.state, _FIELDS) is True  # decided to write; null no-op


def test_stepper_write_control_uses_python_output_flag() -> None:
    hook = FakeFieldHook()
    writer = FieldWriter(write_control=StepperWriteControl(), hook=hook)
    loop = _loop(write_interval=2)
    while loop.run():
        loop.advance()
        writer.write(loop.state, _FIELDS)
    assert len(hook.calls) == 2  # write_time true on indices 2 and 4
