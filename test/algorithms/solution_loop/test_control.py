# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for SolutionControl — the outer-loop predicate that unifies
steady and unsteady, kept separate from the algorithm controls (Pimple/Simple).

Transient (empty residualControl) just advances; steady (residualControl set)
ends the run on residual convergence.
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
    # SolutionControl owns run()/converged()/store_residual(), not the algorithm
    assert not hasattr(pimple, "run")
    assert not hasattr(pimple, "store_residual")


# --- transient: empty residualControl -> never ends early ----------------


def test_transient_advances_to_end_time() -> None:
    sc = SolutionControl()  # empty residualControl
    rt = FakeStepper(end=3.0, dt=1.0)
    steps = 0
    while sc.run(rt):
        rt.increment()
        steps += 1
    assert steps == 3
    assert rt.ended is False


def test_transient_ignores_stored_residuals() -> None:
    sc = SolutionControl()
    sc.store_residual("p", 0.0)
    rt = FakeStepper(end=10.0, dt=1.0)
    assert sc.run(rt) is True
    assert rt.ended is False


# --- steady: residualControl set -> ends on convergence ------------------


def test_steady_does_not_converge_without_residuals() -> None:
    sc = SolutionControl(residualControl={"p": 1e-2, "U": 1e-3})
    rt = FakeStepper(end=10.0, dt=1.0)
    assert sc.run(rt) is True
    assert rt.ended is False  # residuals default to 1.0 -> not converged


def test_steady_converges_and_ends_run() -> None:
    sc = SolutionControl(residualControl={"p": 1e-2, "U": 1e-3})
    sc.store_residual("p", 1e-3)
    sc.store_residual("U", 1e-4)
    rt = FakeStepper(end=10.0, dt=1.0)
    result = sc.run(rt)
    assert rt.ended is True  # converged -> stop() called
    assert result is False


def test_steady_partial_convergence_keeps_running() -> None:
    sc = SolutionControl(residualControl={"p": 1e-2, "U": 1e-3})
    sc.store_residual("p", 1e-3)  # p converged
    sc.store_residual("U", 1e-1)  # U not converged
    rt = FakeStepper(end=10.0, dt=1.0)
    assert sc.run(rt) is True
    assert rt.ended is False
