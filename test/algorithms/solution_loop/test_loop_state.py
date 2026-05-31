# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for how SolutionLoop advances a LoopState — must match Foam::Time.

These tests *are* the advancement specification, derived from
``$FOAM_SRC/OpenFOAM/db/Time/{Time.C,TimeState.C}`` (v2406): ``operator++``,
``run()``, ``end()``, ``setDeltaT``/``adjustDeltaT``. The logic lives in
:class:`SolutionLoop`; the data it mutates is the :class:`LoopState`.
"""

from __future__ import annotations

from typing import Optional

import pytest

from neofoam.algorithms.solution_loop.config import _WRITE_CONTROL_ALIASES
from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.algorithms.solution_loop.solution_loop import SolutionLoop
from neofoam.algorithms.solution_loop.time_integration import (
    SteadyIntegration,
    TimeIntegration,
    TransientIntegration,
)


def _loop(
    *,
    start_time: float = 0.0,
    end_time: float,
    delta_t: float,
    write_control: str = "timeStep",
    write_interval: float = 1.0,
    integration: Optional[TimeIntegration] = None,
) -> SolutionLoop:
    integ = integration if integration is not None else TransientIntegration()
    return SolutionLoop(
        state=LoopState(
            value=start_time,
            delta_t=integ.initial_delta_t(delta_t),
            end_time=end_time,
            start_time=start_time,
            write_control=_WRITE_CONTROL_ALIASES.get(write_control, write_control),
            write_interval=write_interval,
        ),
        integration=integ,
    )


def _drive(loop: SolutionLoop) -> list[tuple[float, float, float, int, bool]]:
    """Run the canonical ``while run(): advance()`` and record per-step state."""
    rows: list[tuple[float, float, float, int, bool]] = []
    while loop.run():
        loop.advance()
        s = loop.state
        rows.append((s.value, s.delta_t, s.delta_t0, s.index, s.write_time))
    return rows


# --- advancement: value / index / stop test ------------------------------


def test_fixed_step_value_and_index_sequence() -> None:
    rows = _drive(_loop(end_time=0.5, delta_t=0.1, write_interval=1))
    assert [r[0] for r in rows] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert [r[3] for r in rows] == [1, 2, 3, 4, 5]


def test_run_and_end_use_half_delta_t_band() -> None:
    loop = _loop(end_time=1.0, delta_t=0.25)
    assert loop.run() is True
    assert loop.end() is False
    while loop.run():
        loop.advance()
    assert loop.run() is False
    assert loop.state.value == pytest.approx(1.0)


# --- old time (deltaT0) bookkeeping ---------------------------------------


def test_delta_t0_is_zero_on_first_step_then_previous_dt() -> None:
    loop = _loop(end_time=1.0, delta_t=0.1)
    loop.advance()
    assert loop.state.delta_t0 == pytest.approx(0.0)  # deltaTSave started at 0
    loop.advance()
    assert loop.state.delta_t0 == pytest.approx(0.1)


def test_delta_t0_tracks_previous_step_across_a_change() -> None:
    loop = _loop(end_time=10.0, delta_t=0.1)
    loop.advance()  # deltaT0=0,   deltaTSave=0.1
    loop.set_delta_t(0.2, adjust=False)
    loop.advance()  # deltaT0=0.1, deltaTSave=0.2
    assert loop.state.delta_t0 == pytest.approx(0.1)
    assert loop.state.delta_t == pytest.approx(0.2)


# --- write control --------------------------------------------------------


def test_write_control_time_step() -> None:
    loop = _loop(end_time=0.4, delta_t=0.1, write_control="timeStep", write_interval=2)
    assert [r[4] for r in _drive(loop)] == [False, True, False, True]


def test_write_control_run_time() -> None:
    loop = _loop(end_time=0.5, delta_t=0.1, write_control="runTime", write_interval=0.2)
    assert [r[4] for r in _drive(loop)] == [False, True, False, True, False]


# --- adjustableRunTime: set_delta_t snaps to land on the write time -------


def test_adjustable_run_time_snaps_delta_t() -> None:
    loop = _loop(
        end_time=1.0, delta_t=0.3, write_control="adjustableRunTime", write_interval=1.0
    )
    loop.set_delta_t(0.3, adjust=True)
    # timeToNextWrite=1.0; nSteps=3.33->round 3; newDeltaT=1/3; >=0.3 so kept (<=2x)
    assert loop.state.delta_t == pytest.approx(1.0 / 3.0)


def test_adjust_no_op_for_time_step_control() -> None:
    loop = _loop(end_time=1.0, delta_t=0.3, write_control="timeStep", write_interval=1)
    loop.set_delta_t(0.3, adjust=True)
    assert loop.state.delta_t == pytest.approx(0.3)


# --- stop() ends the run, without writing ---------------------------------


def test_stop_terminates_run_without_writing() -> None:
    loop = _loop(end_time=10.0, delta_t=1.0)
    assert loop.run() is True
    loop.stop()
    assert loop.run() is False
    # ending the run is not a write step — that is the WriteControl's call
    assert loop.state.write_time is False


# --- low-level advancement sugar ------------------------------------------


def test_set_delta_t_then_advance() -> None:
    loop = _loop(end_time=10.0, delta_t=0.1)
    loop.advance()  # advance one step, unchanged dt
    assert loop.state.index == 1
    loop.set_delta_t(0.2)  # set dt
    loop.advance()
    assert loop.state.delta_t == pytest.approx(0.2)
    assert loop.state.index == 2


# --- iteration (steady) mode ----------------------------------------------


def test_iteration_mode_counts_and_names_as_integers() -> None:
    loop = _loop(
        end_time=3.0, delta_t=1.0, integration=SteadyIntegration(), write_interval=1
    )
    rows = _drive(loop)
    assert [r[3] for r in rows] == [1, 2, 3]
    assert loop.timeName() == "3"


def test_time_mode_time_name_is_general_float() -> None:
    loop = _loop(end_time=1.0, delta_t=0.005)
    loop.advance()
    assert loop.timeName() == "0.005"
