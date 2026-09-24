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

import pytest
from pydantic import ValidationError

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


# --- firstIter / finalIter: where the outer iteration sits in its time step ---
#
# finalIter() is the outer-iteration counterpart of finalInnerIter(); firstIter()
# gates the once-per-time-step mesh move (interFoam moves on
# `pimple.firstIter() || moveMeshOuterCorrectors`). Both are read once per outer
# iteration, so one loop drives both.


@pytest.mark.parametrize(
    ("flag", "nOuterCorrectors", "expected"),
    [
        # nOuterCorrectors defaults to 1 (PISO mode): the single pass is both the
        # first and the final one, exactly like OpenFOAM's pimpleControl.
        ("firstIter", 1, [True]),
        ("finalIter", 1, [True]),
        ("firstIter", 3, [True, False, False]),
        ("finalIter", 3, [False, False, True]),
    ],
)
def test_pimple_iteration_position_flag(
    flag: str, nOuterCorrectors: int, expected: list[bool]
) -> None:
    pimple = PimpleControl(nCorrectors=2, nOuterCorrectors=nOuterCorrectors, momentumPredictor=True)

    flags = []
    while pimple.loop():
        flags.append(getattr(pimple, flag)())

    assert flags == expected


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        # the next time step opens on a first iteration again, and non-final again
        ("firstIter", True),
        ("finalIter", False),
    ],
)
def test_pimple_iteration_position_flag_resets_with_the_loop(flag: str, expected: bool) -> None:
    pimple = PimpleControl(nCorrectors=2, nOuterCorrectors=2, momentumPredictor=True)
    while pimple.loop():
        pass

    assert pimple.loop() is True
    assert getattr(pimple, flag)() is expected


# --- PIMPLE schema accepts real tutorial dicts -----------------------------


def test_pimple_accepts_ncorrectors_one() -> None:
    # Real tutorials (movingCone, propeller, mixerVesselAMI2D, pipeCyclic PIMPLE)
    # all carry `nCorrectors 1`; the schema must accept it (was ge=2, rejected 1).
    control = PimpleControl(
        nOuterCorrectors=2,
        nCorrectors=1,
        nNonOrthogonalCorrectors=0,
        momentumPredictor=True,
        turbCorr=True,
    )
    assert control.nCorrectors == 1

    # a single corrector pass still runs exactly once per outer iteration
    assert control.loop() is True
    corrector_count = 0
    while control.correct():
        corrector_count += 1
    assert corrector_count == 1


def test_pimple_rejects_ncorrectors_zero() -> None:
    # validation is not gutted: nCorrectors must still be >= 1
    with pytest.raises(ValidationError):
        PimpleControl(nCorrectors=0, momentumPredictor=True)


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
    # pEqn.H queries finalInnerIter() *inside* the non-orthogonal loop, and
    # pimpleControl answers `corrPISO_ == nCorrPISO_ && corrNonOrtho_ ==
    # nNonOrthCorr_ + 1` — so only the last non-orthogonal pass of the last
    # corrector picks the `Final` solver settings.
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
    assert control.finalInnerIter() is False
    assert control.correctNonOrthogonal() is True
    assert control.finalNonOrthogonalIter() is True
    assert control.finalInnerIter() is False  # last non-ortho pass, first corrector
    assert control.correctNonOrthogonal() is False

    assert control.correct() is True
    assert control.correctNonOrthogonal() is True
    assert control.finalInnerIter() is False  # last corrector, first non-ortho pass
    assert control.correctNonOrthogonal() is True
    assert control.finalInnerIter() is True


@pytest.mark.parametrize(
    ("nOuterCorrectors", "nCorrectors", "expected"),
    [
        (3, 1, [1, 1, 1]),  # interFoam damBreakPorousBaffle
        (2, 2, [2, 2]),  # interFoam DTCHullMoving
        (1, 2, [2]),  # PISO mode: single outer pass, unchanged
    ],
)
def test_pimple_corrector_restarts_on_every_outer_iteration(
    nOuterCorrectors: int, nCorrectors: int, expected: list[int]
) -> None:
    # pimpleControl::loop() zeroes corrPISO_ at the start of every outer
    # iteration, so each outer corrector runs its own pressure solves
    # (regression: only the first outer iteration ever solved p_rgh).
    control = PimpleControl(
        nOuterCorrectors=nOuterCorrectors,
        nCorrectors=nCorrectors,
        momentumPredictor=True,
    )

    corrector_counts: list[int] = []
    while control.loop():
        corrector_count = 0
        while control.correct():
            corrector_count += 1
        corrector_counts.append(corrector_count)

    assert corrector_counts == expected


def test_pimple_non_ortho_restarts_on_every_outer_iteration() -> None:
    # the reset chain is nested: outer loop re-arms the corrector, which in
    # turn re-arms the non-orthogonal loop on its first iteration.
    control = PimpleControl(
        nOuterCorrectors=2,
        nCorrectors=1,
        nNonOrthogonalCorrectors=1,
        momentumPredictor=True,
    )

    non_ortho_counts: list[int] = []
    while control.loop():
        while control.correct():
            non_ortho_count = 0
            while control.correctNonOrthogonal():
                non_ortho_count += 1
            non_ortho_counts.append(non_ortho_count)

    assert non_ortho_counts == [2, 2]


@pytest.mark.parametrize(
    ("nOuterCorrectors", "nCorrectors", "nNonOrthogonalCorrectors", "expected"),
    [
        # one flag per pressure solve, in solve order (outer x corrector x non-ortho)
        (1, 1, 0, [True]),  # pisoFoam cavity
        (1, 2, 0, [False, True]),  # pimpleFoam RAS/pitzDaily
        (1, 2, 1, [False, False, False, True]),  # pimpleFoam RAS/ellipsekkLOmega
        (1, 1, 2, [False, False, True]),
        # the flag restarts on every outer iteration: it is False again on the
        # first corrector of the second one
        (2, 2, 0, [False, True, False, True]),
        (5, 2, 0, [False, True] * 5),  # pimpleFoam LES/vortexShed
        (2, 2, 1, [False, False, False, True] * 2),
        (15, 1, 0, [True] * 15),  # pimpleFoam laminar/planarContraction
    ],
)
def test_pimple_final_inner_iter_is_last_corrector_and_last_non_ortho(
    nOuterCorrectors: int,
    nCorrectors: int,
    nNonOrthogonalCorrectors: int,
    expected: list[bool],
) -> None:
    # pimpleControl::finalInnerIter() = `corrPISO_ == nCorrPISO_ &&
    # corrNonOrtho_ == nNonOrthCorr_ + 1`: the outer iteration is NOT part of the
    # criterion (that needs finalOnLastPimpleIterOnly), but the non-orthogonal
    # position is — a non-final non-orthogonal pass solves on the loose `p`
    # settings even in the last corrector.
    control = PimpleControl(
        nOuterCorrectors=nOuterCorrectors,
        nCorrectors=nCorrectors,
        nNonOrthogonalCorrectors=nNonOrthogonalCorrectors,
        momentumPredictor=True,
    )

    final_flags: list[bool] = []
    while control.loop():
        while control.correct():
            while control.correctNonOrthogonal():
                final_flags.append(control.finalInnerIter())

    assert final_flags == expected


def test_pimple_final_on_last_pimple_iter_only_restricts_to_the_final_outer_iteration() -> None:
    # `finalOnLastPimpleIterOnly yes` is the one switch that adds finalIter() to
    # the criterion, so only the very last pressure solve of the time step gets
    # the `Final` settings.
    control = PimpleControl(
        nOuterCorrectors=2,
        nCorrectors=2,
        momentumPredictor=True,
        finalOnLastPimpleIterOnly=True,
    )

    final_flags: list[bool] = []
    while control.loop():
        while control.correct():
            while control.correctNonOrthogonal():
                final_flags.append(control.finalInnerIter())

    assert final_flags == [False, False, False, True]


@pytest.mark.parametrize(
    ("turbOnFinalIterOnly", "expected"),
    [
        (True, [False, False, True]),  # native default: once per time step
        (False, [True, True, True]),  # pimpleFoam laminar/planarContraction
    ],
)
def test_pimple_turb_corr_follows_turb_on_final_iter_only(
    turbOnFinalIterOnly: bool, expected: list[bool]
) -> None:
    # pimpleControl::turbCorr() = `!turbOnFinalIterOnly_ || finalIter()`, asked
    # once per outer iteration (pimpleFoam.C's `if (pimple.turbCorr())`).
    control = PimpleControl(
        nOuterCorrectors=3,
        nCorrectors=1,
        momentumPredictor=True,
        turbCorr=True,
        turbOnFinalIterOnly=turbOnFinalIterOnly,
    )

    corrections: list[bool] = []
    while control.loop():
        corrections.append(control.turbCorr())

    assert corrections == expected


def test_pimple_turb_corr_disabled_stays_closed_on_every_iteration() -> None:
    # The `turbCorr no` switch wins over the iteration gate: no correction at all.
    control = PimpleControl(
        nOuterCorrectors=3,
        nCorrectors=1,
        momentumPredictor=True,
        turbCorr=False,
        turbOnFinalIterOnly=False,
    )

    corrections: list[bool] = []
    while control.loop():
        corrections.append(control.turbCorr())

    assert corrections == [False, False, False]


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
