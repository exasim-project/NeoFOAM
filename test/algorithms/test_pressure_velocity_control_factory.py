# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from neofoam.algorithms.control import PimpleControl, SimpleControl


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


def test_simple_flags_are_exposed() -> None:
    control = SimpleControl(
        nNonOrthogonalCorrectors=0,
        momentumPredictor=False,
        consistent=True,
        useResidualConvergence=False,
    )

    assert control.momentumPredictor() is False
    assert control.consistent() is True
