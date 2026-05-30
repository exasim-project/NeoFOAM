# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the pure-Python FoamTime — must match Foam::Time advancement.

These tests *are* the specification, derived from
``$FOAM_SRC/OpenFOAM/db/Time/{Time.C,TimeState.C}`` (v2406): ``operator++``,
``run()``, ``loop()``, ``end()``, ``setDeltaT``/``adjustDeltaT``.
"""

from __future__ import annotations

import pytest

from neofoam.algorithms.foam_time import FoamTime
from neofoam.algorithms.time_integration import SteadyIntegration


def _drive(t: FoamTime) -> list[tuple[float, float, float, int, bool]]:
    """Run the canonical ``while loop():`` and record per-step state."""
    rows: list[tuple[float, float, float, int, bool]] = []
    while t.loop():
        rows.append(
            (
                t.value(),
                t.deltaTValue(),
                t.deltaT0Value(),
                t.timeIndex(),
                t.outputTime(),
            )
        )
    return rows


# --- advancement: value / index / stop test ------------------------------


def test_fixed_step_value_and_index_sequence() -> None:
    t = FoamTime(start_time=0.0, end_time=0.5, delta_t=0.1, write_interval=1)
    rows = _drive(t)
    values = [r[0] for r in rows]
    indices = [r[3] for r in rows]
    assert values == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert indices == [1, 2, 3, 4, 5]


def test_run_and_end_use_half_delta_t_band() -> None:
    t = FoamTime(start_time=0.0, end_time=1.0, delta_t=0.25)
    assert t.run() is True
    assert t.end() is False
    while t.loop():
        pass
    assert t.run() is False
    assert t.value() == pytest.approx(1.0)


# --- old time (deltaT0) bookkeeping ---------------------------------------


def test_delta_t0_is_zero_on_first_step_then_previous_dt() -> None:
    t = FoamTime(start_time=0.0, end_time=1.0, delta_t=0.1)
    t.increment()
    assert t.deltaT0Value() == pytest.approx(0.0)  # deltaTSave started at 0
    t.increment()
    assert t.deltaT0Value() == pytest.approx(0.1)


def test_delta_t0_tracks_previous_step_across_a_change() -> None:
    t = FoamTime(start_time=0.0, end_time=10.0, delta_t=0.1)
    t.increment()  # deltaT0=0,   deltaTSave=0.1
    t.setDeltaT(0.2, adjust=False)
    t.increment()  # deltaT0=0.1, deltaTSave=0.2
    assert t.deltaT0Value() == pytest.approx(0.1)
    assert t.deltaTValue() == pytest.approx(0.2)


# --- write control --------------------------------------------------------


def test_write_control_time_step() -> None:
    t = FoamTime(
        start_time=0.0,
        end_time=0.4,
        delta_t=0.1,
        write_control="timeStep",
        write_interval=2,
    )
    writes = [r[4] for r in _drive(t)]
    assert writes == [False, True, False, True]


def test_write_control_run_time() -> None:
    t = FoamTime(
        start_time=0.0,
        end_time=0.5,
        delta_t=0.1,
        write_control="runTime",
        write_interval=0.2,
    )
    writes = [r[4] for r in _drive(t)]
    assert writes == [False, True, False, True, False]


# --- adjustableRunTime: setDeltaT snaps to land on the write time ---------


def test_adjustable_run_time_snaps_delta_t() -> None:
    t = FoamTime(
        start_time=0.0,
        end_time=1.0,
        delta_t=0.3,
        write_control="adjustableRunTime",
        write_interval=1.0,
    )
    t.setDeltaT(0.3, adjust=True)
    # timeToNextWrite=1.0; nSteps=3.33->round 3; newDeltaT=1/3; >=0.3 so kept (<=2x)
    assert t.deltaTValue() == pytest.approx(1.0 / 3.0)


def test_adjust_no_op_for_time_step_control() -> None:
    t = FoamTime(
        start_time=0.0,
        end_time=1.0,
        delta_t=0.3,
        write_control="timeStep",
        write_interval=1,
    )
    t.setDeltaT(0.3, adjust=True)
    assert t.deltaTValue() == pytest.approx(0.3)


# --- stop() ends the run, without writing ---------------------------------


def test_stop_terminates_run_without_writing() -> None:
    t = FoamTime(start_time=0.0, end_time=10.0, delta_t=1.0)
    assert t.run() is True
    t.stop()
    assert t.run() is False
    # ending the run is not a write step — that is the WriteControl's call
    assert t.outputTime() is False


# --- call-operator sugar --------------------------------------------------


def test_call_operator_advances() -> None:
    t = FoamTime(start_time=0.0, end_time=10.0, delta_t=0.1)
    t()  # advance one step, unchanged dt
    assert t.timeIndex() == 1
    t(0.2)  # set dt then advance
    assert t.deltaTValue() == pytest.approx(0.2)
    assert t.timeIndex() == 2


# --- iteration mode -------------------------------------------------------


def test_iteration_mode_counts_and_names_as_integers() -> None:
    t = FoamTime(
        start_time=0.0,
        end_time=3.0,
        delta_t=1.0,
        integration=SteadyIntegration(),
        write_interval=1,
    )
    rows = _drive(t)
    assert [r[3] for r in rows] == [1, 2, 3]
    assert t.timeName() == "3"


def test_time_mode_time_name_is_general_float() -> None:
    t = FoamTime(start_time=0.0, end_time=1.0, delta_t=0.005)
    t.increment()
    assert t.timeName() == "0.005"
