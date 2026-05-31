# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the FoamTime LoopBackend — mirrors a LoopState onto a backend clock.

Duck-typed on the wrapped time object (setDeltaT + increment), so a fake stands
in for pybFoam.Time and no OpenFOAM is needed.
"""

from __future__ import annotations

from neofoam.algorithms.solution_loop.foam_time import FoamTime
from neofoam.algorithms.solution_loop.loop_state import LoopState


class FakeTime:
    """Stands in for pybFoam.Time at the FoamTime seam (setDeltaT + increment)."""

    def __init__(self) -> None:
        self.delta_t: float = 0.0
        self.steps = 0

    def setDeltaT(self, dt: float) -> None:
        self.delta_t = dt

    def increment(self) -> None:
        self.steps += 1


def _state(*, delta_t: float, index: int) -> LoopState:
    return LoopState(value=0.0, delta_t=delta_t, end_time=1.0, index=index)


def test_satisfies_loop_backend_protocol() -> None:
    from neofoam.algorithms.solution_loop.solution_loop import LoopBackend

    assert isinstance(FoamTime(FakeTime()), LoopBackend)


def test_sets_delta_t_without_advancing_when_index_unchanged() -> None:
    rt = FakeTime()
    backend = FoamTime(rt)
    backend.update(_state(delta_t=0.2, index=0))
    assert rt.delta_t == 0.2
    assert rt.steps == 0  # index still 0 -> no increment


def test_reconciles_backend_index_by_incrementing() -> None:
    rt = FakeTime()
    backend = FoamTime(rt)
    backend.update(_state(delta_t=0.1, index=1))
    backend.update(_state(delta_t=0.1, index=2))
    assert rt.steps == 2  # advanced once per index gap
    assert rt.delta_t == 0.1


def test_catches_up_multiple_steps_in_one_update() -> None:
    rt = FakeTime()
    backend = FoamTime(rt)
    backend.update(_state(delta_t=0.1, index=3))  # jumped 3 indices at once
    assert rt.steps == 3
