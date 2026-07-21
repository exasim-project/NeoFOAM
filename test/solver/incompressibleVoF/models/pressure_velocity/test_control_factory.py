# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Regression test: the VoF PIMPLE control factory reads a real interFoam case.

The five interFoam / interIsoFoam verification cases drive the pressure
correction with ``nCorrectors 1`` (or a frozen-flow single corrector). The
control factory used to hard-fail those with *"the PISO pressure correction
requires nCorrectors >= 2"*; it must instead read the value straight from the
case's ``system/fvSolution`` PIMPLE dict.

The fixture under ``cases/interFoam_damBreakPorousBaffle`` is the unmodified
``system/fvSolution`` copied from
``$FOAM_TUTORIALS/multiphase/interFoam/RAS/damBreakPorousBaffle`` (real OpenFOAM
case file, per TEST_STYLE — never a dict-as-string).
"""

from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.solver.incompressibleVoF.models.pressure_velocity.control_factory import (  # noqa: E402
    create_pimple_control,
)

_CASES = Path(__file__).parent / "cases"


def test_real_interfoam_case_with_single_corrector_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real damBreakPorousBaffle case (nCorrectors 1) builds the control
    instead of raising the old ``nCorrectors >= 2`` error."""
    monkeypatch.chdir(_CASES / "interFoam_damBreakPorousBaffle")
    control = create_pimple_control({})
    assert control.nCorrectors == 1
    assert control.nOuterCorrectors == 3
    assert control.momentumPredictor() is False
