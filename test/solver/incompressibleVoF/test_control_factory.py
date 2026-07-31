# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the incompressibleVoF PIMPLE control factory.

The factory reads the ``PIMPLE`` subdict of ``system/fvSolution`` from the
cwd; each test chdirs into a real case directory under ``cases/`` (per the
convention that dict-reading tests load real OpenFOAM case files).
"""

from pathlib import Path

import pytest

from neofoam.solver.incompressibleVoF.models.pressure_velocity.control_factory import (
    FrozenFlowControl,
    create_pimple_control,
)

_CASES = Path(__file__).parent / "cases"


def test_missing_ncorrectors_defaults_to_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent nCorrectors falls back to 2 — the smallest valid PISO count."""
    monkeypatch.chdir(_CASES / "pimple_defaults")
    control = create_pimple_control({})
    assert control.nCorrectors == 2
    assert control.momentumPredictor() is False


def test_ncorrectors_one_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """nCorrectors 1 is read straight from the case dict (real interFoam cases
    drive the pressure correction with a single corrector), not rejected."""
    monkeypatch.chdir(_CASES / "pimple_one_corrector")
    control = create_pimple_control({})
    assert control.nCorrectors == 1


def test_all_pimple_keys_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_CASES / "pimple_full")
    control = create_pimple_control({})
    assert control.nOuterCorrectors == 2
    assert control.nCorrectors == 3
    assert control.nNonOrthogonalCorrectors == 1
    assert control.momentumPredictor() is False
    assert control.turbCorr() is False


def test_turb_on_final_iter_only_defaults_to_correcting_on_the_last_outer_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pimpleControl::read()'s default is ``turbOnFinalIterOnly true``.

    ``cases/pimple_outer3`` leaves the key out, as every interFoam tutorial
    does, so the correction has to land on the third outer corrector alone.
    """
    monkeypatch.chdir(_CASES / "pimple_outer3")
    control = create_pimple_control({})
    assert control.turbOnFinalIterOnly is True
    assert [control.loop() and control.turbCorr() for _ in range(3)] == [False, False, True]


def test_turb_on_final_iter_only_no_puts_the_correction_in_every_outer_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "pimple_turb_every_outer")
    control = create_pimple_control({})
    assert control.turbOnFinalIterOnly is False
    assert [control.loop() and control.turbCorr() for _ in range(3)] == [True, True, True]


def test_a_frozen_flow_case_never_corrects_the_turbulence() -> None:
    # interIsoFoam ``continue``s out of the outer corrector before the
    # turbulence correction, so the frozen-flow control answers no.
    assert FrozenFlowControl().turbCorr() is False
