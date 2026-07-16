# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the controlDict time config and its advancement helpers.

``TimeControlConfig`` is the validated ``system/controlDict`` slice the loop loads;
the ``_WRITE_CONTROL_ALIASES`` table and ``_round_half_away`` helper encode the
Foam::Time advancement semantics the engine relies on. The load test reads a real
checked-in case (the same fixtures the parity test drives), never a dict-as-string.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neofoam.algorithms.solution_loop.config import (
    _ADJUSTABLE,
    _RUN_TIME,
    _TIME_STEP,
    _WRITE_CONTROL_ALIASES,
    TimeControlConfig,
    _round_half_away,
)

_CASES = Path(__file__).parent / "cases"


# --- write-control alias table -------------------------------------------


@pytest.mark.parametrize(
    ("keyword", "canonical"),
    [
        ("timeStep", _TIME_STEP),
        ("runTime", _RUN_TIME),
        ("adjustableRunTime", _ADJUSTABLE),
        ("adjustable", _ADJUSTABLE),  # the neofoam-only alias folds onto the same kind
    ],
)
def test_write_control_alias_maps_to_canonical_kind(
    keyword: str, canonical: str
) -> None:
    assert _WRITE_CONTROL_ALIASES[keyword] == canonical


# --- Foam::round (half away from zero, not banker's rounding) -------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.5, 1), (1.5, 2), (2.5, 3), (-0.5, -1), (-1.5, -2), (3.33, 3)],
)
def test_round_half_away_from_zero(value: float, expected: int) -> None:
    # Python's built-in round() is banker's rounding: round(2.5) == 2. Foam::round
    # (and this helper) rounds halves away from zero, so 2.5 -> 3.
    assert _round_half_away(value) == expected


# --- TimeControlConfig validation + load ---------------------------------


def test_defaults_come_from_write_control_config() -> None:
    config = TimeControlConfig(endTime=1.0, deltaT=0.1)
    assert config.writeControl == "timeStep"
    assert config.writeInterval == 1.0
    assert config.startTime == 0.0


@pytest.mark.parametrize(
    "bad", [{"endTime": 0.0, "deltaT": 0.1}, {"endTime": 1.0, "deltaT": 0.0}]
)
def test_end_time_and_delta_t_must_be_positive(bad: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        TimeControlConfig(**bad)


def test_loads_a_real_control_dict() -> None:
    config = TimeControlConfig.load(case_dir=_CASES / "runTime")
    assert config.writeControl == "runTime"
    assert config.writeInterval == pytest.approx(0.2)
    assert config.endTime == pytest.approx(0.5)
    assert config.deltaT == pytest.approx(0.1)
