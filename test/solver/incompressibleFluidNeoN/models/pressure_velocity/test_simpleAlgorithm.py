# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The SIMPLE control block, read into the per-run loop state without a case.

``_build_simple_state`` is everything ``create_simple_state`` does once the
runtime has handed it the converted ``system/fvSolution``: the SIMPLEC switch
(``consistent``) that selects the rAtU pressure correction, the non-orthogonal
corrector count, and the momentumPredictor switch. It is a pure function of a
``NeoN::Dictionary``, so it is exercised in-process (see
``test_pimpleAlgorithm`` for why the dictionaries are built rather than read).

The entries mirror the SIMPLEC ``fvSolution`` of
``cases/pitzDailySteady/solution/simplec``. That the consistent branch is
numerically right — not merely selected — is a separate, heavier check:
``test_steady_vs_incompressibleFluid`` compares a SIMPLEC run against the
pybFoam backend field by field.
"""

from __future__ import annotations

from typing import Any, Union

import neon._neon as nn  # NeoN Python bindings
import pytest

from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.simpleAlgorithm import (
    _build_simple_state,
)


def _fv_solution(**simple_entries: Union[int, str]) -> Any:
    """An fvSolution dictionary whose SIMPLE block carries ``simple_entries``."""
    simple = nn.Dictionary()
    for key, value in simple_entries.items():
        if isinstance(value, int):
            simple.insert_int(key, value)
        else:
            simple.insert_string(key, value)
    fv_solution = nn.Dictionary()
    fv_solution.insert_dict("SIMPLE", simple)
    return fv_solution


def _non_orthogonal_passes(piso: Any) -> int:
    """How many times the non-orthogonal corrector loop runs before it resets."""
    return sum(1 for _ in iter(piso.correct_non_orthogonal, False))


@pytest.mark.parametrize(
    ("written", "expected"),
    [("yes", True), ("no", False), ("on", True), ("1", True)],
)
def test_simple_state_reads_the_consistent_switch(written: str, expected: bool) -> None:
    """``consistent`` selects (or does not select) the SIMPLEC correction."""
    fv_solution = _fv_solution(consistent=written)

    state = _build_simple_state(fv_solution)

    assert state.consistent is expected


def test_simple_state_without_consistent_is_plain_simple() -> None:
    """A case that omits ``consistent`` takes the rAU (non-SIMPLEC) path."""
    state = _build_simple_state(_fv_solution())

    assert state.consistent is False


def test_simple_state_reads_the_non_orthogonal_corrector_count() -> None:
    """``nNonOrthogonalCorrectors 2`` gives three pressure-correction passes."""
    fv_solution = _fv_solution(nNonOrthogonalCorrectors=2)

    state = _build_simple_state(fv_solution)

    assert _non_orthogonal_passes(state.piso) == 3


def test_simple_state_without_non_orthogonal_correctors_makes_one_pass() -> None:
    """The OpenFOAM default (0 correctors) is a single pressure-correction pass."""
    state = _build_simple_state(_fv_solution())

    assert _non_orthogonal_passes(state.piso) == 1


def test_simple_state_defaults_the_momentum_predictor_on() -> None:
    """solutionControl defaults momentumPredictor to true; SIMPLE follows it."""
    state = _build_simple_state(_fv_solution())

    assert state.piso.momentum_predictor() is True
