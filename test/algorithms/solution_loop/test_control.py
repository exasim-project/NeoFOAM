# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for SolutionControl — the outer-loop predicate, kept separate from
the algorithm controls (Pimple/Simple).

SolutionControl governs advancement only; stopping (residual convergence and every
other criterion) flows through the loopCondition fold, not this control.
"""

from __future__ import annotations

from neofoam.algorithms.solution_loop.control import PimpleControl, SolutionControl


class FakeStepper:
    def __init__(self, *, end: float = 5.0, dt: float = 1.0) -> None:
        self._t = 0.0
        self._dt = dt
        self._end = end
        self._ended = False

    def run(self) -> bool:
        return (not self._ended) and self._t < self._end - 1e-10

    def increment(self) -> None:
        self._t = min(self._t + self._dt, self._end)

    def stop(self) -> None:
        self._ended = True

    @property
    def ended(self) -> bool:
        return self._ended


# --- separation of concerns ----------------------------------------------


def test_algorithm_control_has_no_solution_loop_methods() -> None:
    pimple = PimpleControl(nCorrectors=2, momentumPredictor=True)
    # SolutionControl owns run(), not the algorithm control.
    assert not hasattr(pimple, "run")


def test_solution_control_has_no_residual_convergence_path() -> None:
    sc = SolutionControl()
    assert not hasattr(sc, "converged")
    assert not hasattr(sc, "store_residual")
    assert not hasattr(sc, "residualControl")


# --- advancement only -----------------------------------------------------


def test_transient_advances_to_end_time() -> None:
    sc = SolutionControl()
    rt = FakeStepper(end=3.0, dt=1.0)
    steps = 0
    while sc.run(rt):
        rt.increment()
        steps += 1
    assert steps == 3
    assert rt.ended is False
