# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the ``WriteControl`` policies and factory."""

from __future__ import annotations

from neofoam.algorithms.field_writer.write_control import (
    StepperWriteControl,
    IntervalWriteControl,
    RunTimeWriteControl,
    WriteControl,
    WriteControlConfig,
    write_control_from_config,
)


class FakeStepper:
    """Minimal StepView: the attributes WriteControl reads (value/index/write_time)."""

    def __init__(self, *, end: float = 1.0, dt: float = 1.0) -> None:
        self.value = 0.0
        self._dt = dt
        self._end = end
        self.index = 0
        self.write_time = False

    def run(self) -> bool:
        return self.value < self._end - 1e-10

    def increment(self) -> None:
        self.value = round(self.value + self._dt, 10)
        self.index += 1


def test_policies_satisfy_protocol() -> None:
    assert isinstance(IntervalWriteControl(interval=1), WriteControl)
    assert isinstance(RunTimeWriteControl(interval=1.0), WriteControl)
    assert isinstance(StepperWriteControl(), WriteControl)


def test_interval_writes_every_n_steps() -> None:
    wc = IntervalWriteControl(interval=2)
    rt = FakeStepper(end=4.0, dt=1.0)
    decisions = []
    while rt.run():
        rt.increment()
        decisions.append(wc.should_write(rt))
    # indices 1,2,3,4 -> write on the even ones
    assert decisions == [False, True, False, True]


def test_runtime_writes_every_interval_of_sim_time() -> None:
    wc = RunTimeWriteControl(interval=0.2, start=0.0)
    rt = FakeStepper(end=0.5, dt=0.1)
    decisions = []
    while rt.run():
        rt.increment()
        decisions.append(wc.should_write(rt))
    # times 0.1,0.2,0.3,0.4,0.5 -> write when >= 0.2, 0.4
    assert decisions == [False, True, False, True, False]


def test_stepper_write_control_forwards_write_flag() -> None:
    writing = FakeStepper()
    writing.write_time = True
    assert StepperWriteControl().should_write(writing) is True
    assert StepperWriteControl().should_write(FakeStepper()) is False


def test_factory_timestep() -> None:
    wc = write_control_from_config(
        WriteControlConfig(writeControl="timeStep", writeInterval=5)
    )
    assert isinstance(wc, IntervalWriteControl)


def test_factory_runtime() -> None:
    wc = write_control_from_config(
        WriteControlConfig(writeControl="adjustable", writeInterval=1.0)
    )
    assert isinstance(wc, RunTimeWriteControl)


def test_factory_stepper_decides() -> None:
    wc = write_control_from_config(WriteControlConfig(), stepper_decides=True)
    assert isinstance(wc, StepperWriteControl)


def test_factory_routes_values_through_the_discriminated_union() -> None:
    # selection goes through WriteControl.create(policy=...); the controlDict
    # write keys must arrive on the chosen policy's fields.
    wc = write_control_from_config(
        WriteControlConfig(writeControl="runTime", writeInterval=0.25, startTime=2.0)
    )
    assert isinstance(wc, RunTimeWriteControl)
    assert wc.interval == 0.25
    assert wc.start == 2.0
