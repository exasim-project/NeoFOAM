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

``cases/interIsoFoam_discInConstantFlow`` is the unmodified ``system/fvSolution``
from ``$FOAM_TUTORIALS/multiphase/interIsoFoam/discInConstantFlow``: the
``frozenFlow yes`` idiom with the deliberate ``nCorrectors -1`` /
``nNonOrthogonalCorrectors -1`` sentinels. The factory must skip the
pressure-velocity solve for it (a :class:`FrozenFlowControl`, no
:class:`PimpleControl`) rather than reject the negative counts.
"""

from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from pydantic import ValidationError  # noqa: E402

from neofoam.algorithms.solution_loop.control import PimpleControl  # noqa: E402
from neofoam.solver.incompressibleVoF.models.pressure_velocity.control_factory import (  # noqa: E402
    FrozenFlowControl,
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
    assert isinstance(control, PimpleControl)
    assert control.nCorrectors == 1
    assert control.nOuterCorrectors == 3
    assert control.momentumPredictor() is False


def test_frozen_flow_skips_pressure_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """frozenFlow yes with the -1 sentinel counts builds a FrozenFlowControl
    (no PimpleControl, no ValidationError) that switches the solve off."""
    monkeypatch.chdir(_CASES / "interIsoFoam_discInConstantFlow")
    control = create_pimple_control({})
    # No PimpleControl is built, so the -1 sentinels never hit its ge=1 bound.
    assert isinstance(control, FrozenFlowControl)
    # The outer loop still runs exactly one pass (so alpha advection runs once
    # per step), then closes and re-arms for the next time step.
    assert control.loop() is True
    assert control.finalIter() is True
    assert control.loop() is False
    assert control.loop() is True


def test_frozen_flow_no_with_negative_correctors_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative corrector counts WITHOUT frozenFlow are a real misconfiguration:
    the ge=1/ge=0 bounds must still reject them (the fix must not mask typos)."""
    monkeypatch.chdir(_CASES / "pimple_frozenflow_no_negative_correctors")
    with pytest.raises(ValidationError):
        create_pimple_control({})
