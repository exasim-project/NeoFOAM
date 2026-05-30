# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the FieldWriter — persisting fields, the symmetric twin of the loop.

``Writer`` is the abstraction (implement it to add new writers); ``FieldWriter``
is the concrete one for simulation fields. Like :class:`SolutionLoop` (advance +
push steps to a StepSink), it owns a WriteControl *policy* (when to write) and
writes through a :class:`FieldHook` *backend* (the action). The decision reads
the pure-Python stepper; the action runs only on write steps.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import Field

from neofoam.algorithms.foam_time import FoamTime
from neofoam.algorithms.write_control import IntervalWriteControl, StepperWriteControl
from neofoam.algorithms.writer import FieldHook, FieldWriter, Writer


class FakeFieldHook(FieldHook):
    """Records the fields handed to it on each write (a fake backend)."""

    calls: list[list[str]] = Field(default_factory=list)

    def write_fields(self, fields: Mapping[str, Any]) -> None:
        self.calls.append(sorted(fields))


_FIELDS: dict[str, Any] = {"U": object(), "p": object()}


def _stepper(**kw: object) -> FoamTime:
    base: dict[str, object] = {"end_time": 0.4, "delta_t": 0.1}
    base.update(kw)
    return FoamTime(**base)  # type: ignore[arg-type]


def test_field_writer_is_a_writer() -> None:
    assert isinstance(FieldWriter(write_control=StepperWriteControl()), Writer)


def test_writes_fields_on_write_steps_only() -> None:
    hook = FakeFieldHook()
    writer = FieldWriter(write_control=IntervalWriteControl(interval=2), hook=hook)
    stepper = _stepper()
    while stepper.loop():
        writer.write(stepper, _FIELDS)
    # indices 1,2,3,4 -> write on 2 and 4, each time handed all the fields
    assert hook.calls == [["U", "p"], ["U", "p"]]


def test_write_returns_whether_it_wrote() -> None:
    hook = FakeFieldHook()
    writer = FieldWriter(write_control=StepperWriteControl(), hook=hook)
    stepper = _stepper(end_time=0.2, delta_t=0.1, write_interval=1)
    stepper.increment()  # index 1, a write step (writeInterval 1)
    assert writer.write(stepper, _FIELDS) is True
    assert hook.calls == [["U", "p"]]


def test_should_write_does_not_write() -> None:
    hook = FakeFieldHook()
    writer = FieldWriter(write_control=IntervalWriteControl(interval=2), hook=hook)
    stepper = _stepper()
    stepper.increment()  # index 1 -> not a write step for interval 2
    assert writer.should_write(stepper) is False
    assert hook.calls == []


def test_null_hook_is_the_default() -> None:
    # no hook -> decision still works, nothing is persisted (standalone use)
    writer = FieldWriter(write_control=StepperWriteControl())
    stepper = _stepper(write_interval=1)
    stepper.increment()
    assert writer.write(stepper, _FIELDS) is True  # decided to write; null hook no-op


def test_stepper_write_control_uses_python_output_flag() -> None:
    hook = FakeFieldHook()
    writer = FieldWriter(write_control=StepperWriteControl(), hook=hook)
    stepper = _stepper(end_time=0.4, delta_t=0.1, write_interval=2)
    while stepper.loop():
        writer.write(stepper, _FIELDS)
    assert len(hook.calls) == 2  # outputTime true on indices 2 and 4
