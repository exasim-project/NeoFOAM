# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the incompressibleFluid PIMPLE control factory.

The factory reads the ``PIMPLE`` (or, for pisoFoam cases, ``PISO``) subdict of
``system/fvSolution`` from the cwd; each test chdirs into a real OpenFOAM case
directory under ``cases/`` (per the convention that dict-reading tests load
real case files).
"""

from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.algorithms.solution_loop.control import PimpleControl  # noqa: E402
from neofoam.solver.incompressibleFluid.models.pressure_velocity.control_factory import (  # noqa: E402
    create_pimple_control,
)

_CASES = Path(__file__).parent / "cases"


def test_piso_block_maps_onto_pimple_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pisoFoam PISO block builds a PimpleControl (nOuterCorrectors fixed to 1)."""
    monkeypatch.chdir(_CASES / "piso_cavity")
    control = create_pimple_control({})
    assert isinstance(control, PimpleControl)
    assert control.nOuterCorrectors == 1
    assert control.nCorrectors == 2
    assert control.nNonOrthogonalCorrectors == 0
    assert control.momentumPredictor() is True


def test_pimple_block_still_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing PIMPLE path keeps reading all keys."""
    monkeypatch.chdir(_CASES / "pimple_full")
    control = create_pimple_control({})
    assert control.nOuterCorrectors == 2
    assert control.nCorrectors == 3
    assert control.nNonOrthogonalCorrectors == 1
    assert control.momentumPredictor() is False
    assert control.turbCorr() is False
    # pimpleControl::read() defaults for the two keys the dict leaves out
    assert control.turbOnFinalIterOnly is True
    assert control.finalOnLastPimpleIterOnly is False


def test_turbulence_correction_switches_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """`turbOnFinalIterOnly no` opens the correction gate in every outer iteration."""
    monkeypatch.chdir(_CASES / "pimple_turb_every_iteration")
    control = create_pimple_control({})
    assert control.turbOnFinalIterOnly is False
    assert control.finalOnLastPimpleIterOnly is True

    corrections = []
    while control.loop():
        corrections.append(control.turbCorr())
    assert corrections == [True, True, True]
