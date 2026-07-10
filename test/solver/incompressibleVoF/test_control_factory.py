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
    create_pimple_control,
)

_CASES = Path(__file__).parent / "cases"


def test_missing_ncorrectors_defaults_to_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent nCorrectors falls back to 2 — the smallest valid PISO count."""
    monkeypatch.chdir(_CASES / "pimple_defaults")
    control = create_pimple_control({})
    assert control.nCorrectors == 2
    assert control.momentumPredictor() is False


def test_ncorrectors_one_is_rejected_with_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nCorrectors 1 fails with a solver-level message (PISO needs >= 2),
    not a raw Pydantic ValidationError."""
    monkeypatch.chdir(_CASES / "pimple_one_corrector")
    with pytest.raises(ValueError, match="nCorrectors >= 2"):
        create_pimple_control({})


def test_all_pimple_keys_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_CASES / "pimple_full")
    control = create_pimple_control({})
    assert control.nOuterCorrectors == 2
    assert control.nCorrectors == 3
    assert control.nNonOrthogonalCorrectors == 1
    assert control.momentumPredictor() is False
    assert control.turbCorr() is False
