# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the control bundles in ``algorithms.solution_loop.control``.

SolutionControl — the outer-loop predicate that unifies steady and unsteady,
kept separate from the algorithm controls (Pimple/Simple). Transient (empty
residualControl) just advances; steady (residualControl set) ends the run on
residual convergence.

PimpleControl / SimpleControl — the corrector-loop mechanics the solvers'
control factories build (nested corrector / non-orthogonal counting, final-iter
flags, per-time-step reset).
"""

from __future__ import annotations

from neofoam.algorithms.solution_loop.control import (
    PimpleControl,
    SimpleControl,
    SolutionControl,
)


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


# --- finalIter: the outer-iteration counterpart of finalInnerIter ---------


def test_pimple_final_iter_piso_mode_is_always_final() -> None:
    # nOuterCorrectors defaults to 1 (PISO mode): every outer iteration is
    # the final one, exactly like OpenFOAM's pimpleControl.
    pimple = PimpleControl(nCorrectors=2, momentumPredictor=True)
    assert pimple.loop() is True
    assert pimple.finalIter() is True
    assert pimple.loop() is False


def test_pimple_final_iter_true_only_on_last_outer_corrector() -> None:
    pimple = PimpleControl(nCorrectors=2, nOuterCorrectors=3, momentumPredictor=True)
    finals = []
    while pimple.loop():
        finals.append(pimple.finalIter())
    assert finals == [False, False, True]


def test_pimple_final_iter_resets_with_the_loop() -> None:
    pimple = PimpleControl(nCorrectors=2, nOuterCorrectors=2, momentumPredictor=True)
    while pimple.loop():
        pass
    # next time step starts non-final again
    assert pimple.loop() is True
    assert pimple.finalIter() is False


# --- PIMPLE corrector / non-orthogonal loop mechanics ----------------------


def test_pimple_corrector_and_non_ortho_counts() -> None:
    control = PimpleControl(
        nOuterCorrectors=1,
        nCorrectors=2,
        nNonOrthogonalCorrectors=1,
        momentumPredictor=True,
        turbCorr=True,
    )

    assert control.loop() is True

    corrector_count = 0
    non_ortho_counts: list[int] = []

    while control.correct():
        corrector_count += 1

        non_ortho_count = 0
        while control.correctNonOrthogonal():
            non_ortho_count += 1
        non_ortho_counts.append(non_ortho_count)

    assert corrector_count == 2
    # non-ortho counter resets on every corrector iteration (linked condition)
    assert non_ortho_counts == [2, 2]


def test_pimple_final_flags_on_last_iterations() -> None:
    control = PimpleControl(
        nOuterCorrectors=1,
        nCorrectors=2,
        nNonOrthogonalCorrectors=1,
        momentumPredictor=True,
        turbCorr=True,
    )

    assert control.loop() is True

    assert control.correct() is True
    assert control.finalInnerIter() is False

    assert control.correctNonOrthogonal() is True
    assert control.finalNonOrthogonalIter() is False
    assert control.correctNonOrthogonal() is True
    assert control.finalNonOrthogonalIter() is True
    assert control.correctNonOrthogonal() is False

    assert control.correct() is True
    assert control.finalInnerIter() is True


# --- SIMPLE loop mechanics --------------------------------------------------


def test_simple_single_inner_pass_and_non_ortho_count() -> None:
    control = SimpleControl(
        nNonOrthogonalCorrectors=2,
        momentumPredictor=True,
        consistent=False,
        useResidualConvergence=False,
    )

    assert control.loop() is True
    assert control.loop() is False

    assert control.loop() is True
    non_ortho_count = 0
    while control.correctNonOrthogonal():
        non_ortho_count += 1

    assert non_ortho_count == 3

    # Closing the pass re-arms the non-orthogonal corrector: the next outer
    # iteration must run its pressure solves again (regression: only the
    # first SIMPLE iteration ever solved p).
    assert control.loop() is False
    assert control.loop() is True
    non_ortho_count = 0
    while control.correctNonOrthogonal():
        non_ortho_count += 1
    assert non_ortho_count == 3


def test_simple_flags_are_exposed() -> None:
    control = SimpleControl(
        nNonOrthogonalCorrectors=0,
        momentumPredictor=False,
        consistent=True,
        useResidualConvergence=False,
    )

    assert control.momentumPredictor() is False
    assert control.consistent() is True
