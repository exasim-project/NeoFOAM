# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for injectable deltaT stability constraints + aggregation.

The next deltaT is the minimum over every constraint models inject (CFL, a VoF
interface Courant, a maxDeltaT cap, …), then a growth clamp.
"""

from __future__ import annotations

import pytest

from neofoam.algorithms.constraints.time_step import (
    VGREAT,
    CourantConstraint,
    DeltaTConstraint,
    MaxDeltaTConstraint,
    next_delta_t,
)


class _Ctx:
    """Minimal context a constraint reads: current Courant + current deltaT."""

    def __init__(self, *, co: float, dt: float) -> None:
        self._co = co
        self._dt = dt

    def max_courant(self) -> float:
        return self._co

    def current_delta_t(self) -> float:
        return self._dt


def test_constraints_satisfy_protocol() -> None:
    assert isinstance(MaxDeltaTConstraint(maxDeltaT=1.0), DeltaTConstraint)
    assert isinstance(CourantConstraint(maxCo=0.5), DeltaTConstraint)


def test_max_delta_t_constraint_returns_cap() -> None:
    assert MaxDeltaTConstraint(maxDeltaT=2.0).max_delta_t(_Ctx(co=1.0, dt=0.1)) == 2.0


def test_courant_constraint_scales_with_courant() -> None:
    # allowed = maxCo/Co * dt = 0.5/1.0 * 0.1
    assert CourantConstraint(maxCo=0.5).max_delta_t(
        _Ctx(co=1.0, dt=0.1)
    ) == pytest.approx(0.05)


def test_courant_constraint_no_limit_when_co_zero() -> None:
    assert CourantConstraint(maxCo=0.5).max_delta_t(_Ctx(co=0.0, dt=0.1)) == VGREAT


def test_next_delta_t_takes_min_over_injected_constraints() -> None:
    ctx = _Ctx(co=1.0, dt=0.1)
    cons: list[DeltaTConstraint] = [
        CourantConstraint(maxCo=0.5),
        MaxDeltaTConstraint(maxDeltaT=1.0),
    ]
    # Courant allows 0.05, cap allows 1.0 -> min 0.05 (growth clamp 0.12 not binding)
    assert next_delta_t(cons, ctx, current_dt=0.1) == pytest.approx(0.05)


def test_next_delta_t_growth_is_capped() -> None:
    ctx = _Ctx(co=0.1, dt=0.1)  # Courant allows 0.5/0.1*0.1 = 0.5
    cons = [CourantConstraint(maxCo=0.5)]
    assert next_delta_t(cons, ctx, current_dt=0.1) == pytest.approx(0.12)  # 1.2 x 0.1


def test_next_delta_t_cap_can_bind_under_growth() -> None:
    ctx = _Ctx(co=0.1, dt=1.0)  # Courant allows 5.0; cap 1.1; growth 1.2
    cons: list[DeltaTConstraint] = [
        CourantConstraint(maxCo=0.5),
        MaxDeltaTConstraint(maxDeltaT=1.1),
    ]
    assert next_delta_t(cons, ctx, current_dt=1.0) == pytest.approx(1.1)


def test_next_delta_t_no_constraints_keeps_current() -> None:
    assert next_delta_t([], _Ctx(co=1.0, dt=0.1), current_dt=0.1) == 0.1
