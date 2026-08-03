# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the incompressibleVoF PIMPLE control factory.

The factory reads the ``PIMPLE`` subdict of ``system/fvSolution`` from the cwd,
so each test chdirs into a real case directory under
``test/solver/incompressibleVoF/cases/`` (dict-reading tests load real OpenFOAM
case files, never a dict-as-string). Those cases live one package up because
``test_turbulence_correction.py`` drives the same controls through the solver's
turbulence step.

Two of them are unmodified upstream tutorial files, and they are what the
regression in this module is about — the five interFoam / interIsoFoam
verification cases drive the pressure correction with ``nCorrectors 1`` (or a
frozen-flow single corrector), which the factory used to hard-fail with *"the
PISO pressure correction requires nCorrectors >= 2"*:

* ``cases/interFoam_damBreakPorousBaffle`` — ``system/fvSolution`` from
  ``$FOAM_TUTORIALS/multiphase/interFoam/RAS/damBreakPorousBaffle``
  (``nCorrectors 1``);
* ``cases/interIsoFoam_discInConstantFlow`` — from
  ``$FOAM_TUTORIALS/multiphase/interIsoFoam/discInConstantFlow``: the
  ``frozenFlow yes`` idiom with the deliberate ``nCorrectors -1`` /
  ``nNonOrthogonalCorrectors -1`` sentinels. The factory must skip the
  pressure-velocity solve for it (a :class:`FrozenFlowControl`, no
  :class:`PimpleControl`) rather than reject the negative counts.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from neofoam.algorithms.solution_loop.control import PimpleControl
from neofoam.solver.incompressibleVoF.models.pressure_velocity.control_factory import (
    FrozenFlowControl,
    create_pimple_control,
)

_CASES = Path(__file__).parents[2] / "cases"


def test_missing_ncorrectors_defaults_to_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent nCorrectors falls back to 2 — the smallest valid PISO count."""
    monkeypatch.chdir(_CASES / "pimple_defaults")
    control = create_pimple_control({})
    assert control.nCorrectors == 2
    assert control.momentumPredictor() is False


def test_all_pimple_keys_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_CASES / "pimple_full")
    control = create_pimple_control({})
    assert control.nOuterCorrectors == 2
    assert control.nCorrectors == 3
    assert control.nNonOrthogonalCorrectors == 1
    assert control.momentumPredictor() is False
    assert control.turbCorr() is False


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


@pytest.mark.parametrize(
    "case_name, turb_on_final_iter_only, expected_turb_corr",
    [
        # pimpleControl::read()'s default is ``turbOnFinalIterOnly true``, and
        # cases/pimple_outer3 leaves the key out as every interFoam tutorial
        # does — so the correction has to land on the third outer corrector alone.
        pytest.param("pimple_outer3", True, [False, False, True], id="default"),
        # The same case with ``turbOnFinalIterOnly no``.
        pytest.param("pimple_turb_every_outer", False, [True, True, True], id="no"),
    ],
)
def test_turb_on_final_iter_only_places_the_correction_where_the_case_asks(
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    turb_on_final_iter_only: bool,
    expected_turb_corr: list[bool],
) -> None:
    monkeypatch.chdir(_CASES / case_name)
    control = create_pimple_control({})
    assert control.turbOnFinalIterOnly is turb_on_final_iter_only
    assert [control.loop() and control.turbCorr() for _ in range(3)] == expected_turb_corr


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


def test_a_frozen_flow_case_never_corrects_the_turbulence() -> None:
    # interIsoFoam ``continue``s out of the outer corrector before the
    # turbulence correction, so the frozen-flow control answers no.
    assert FrozenFlowControl().turbCorr() is False


def test_frozen_flow_no_with_negative_correctors_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative corrector counts WITHOUT frozenFlow are a real misconfiguration:
    the ge=1/ge=0 bounds must still reject them (the fix must not mask typos)."""
    monkeypatch.chdir(_CASES / "pimple_frozenflow_no_negative_correctors")
    with pytest.raises(ValidationError):
        create_pimple_control({})
