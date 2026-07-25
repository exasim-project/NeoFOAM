# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the incompressibleVoF pressure-velocity dispatcher (``base.py``).

VoF always uses PIMPLE (interFoam has no SIMPLE/PISO support), so unlike the
incompressibleFluid dispatcher this one only ever has a single family member.
``all_specs`` is case-free; ``detect_and_create`` reads ``system/fvSolution``
and warns (but still returns the PIMPLE spec) when no PIMPLE dict is found
there.

The "missing PIMPLE dict" case is derived from the real damBreakPorousBaffle
fvSolution at test time via pybFoam's own ``dictionary.read``/``remove``/
``write`` (TEST_STYLE rule 3: vary content through the format's reader/writer,
never text-patch a copy) since every real interFoam tutorial case does carry
a PIMPLE dict.
"""

import shutil
from pathlib import Path

import pytest
from pybFoam import dictionary

from neofoam.solver.incompressibleVoF.models.pressure_velocity.base import (
    PressureVelocityAlgorithm,
)
from neofoam.solver.incompressibleVoF.models.pressure_velocity.pimpleAlgorithm import (
    pimple,
)

_CASES = Path(__file__).parent / "cases"


def test_all_specs_returns_only_the_pimple_spec() -> None:
    assert PressureVelocityAlgorithm.all_specs() == [pimple]


def test_detect_and_create_returns_pimple_for_the_real_interfoam_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "interFoam_damBreakPorousBaffle")
    assert PressureVelocityAlgorithm.detect_and_create() is pimple


def test_detect_and_create_warns_and_still_returns_pimple_without_a_PIMPLE_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real damBreakPorousBaffle fvSolution, with its PIMPLE dict removed
    via pybFoam's own reader/writer, has no PIMPLE dict at all."""
    shutil.copytree(_CASES / "interFoam_damBreakPorousBaffle", tmp_path, dirs_exist_ok=True)
    fv_solution_path = tmp_path / "system" / "fvSolution"
    fv_solution = dictionary.read(str(fv_solution_path))
    fv_solution.remove("PIMPLE")
    fv_solution.write(str(fv_solution_path))

    monkeypatch.chdir(tmp_path)
    with pytest.warns(UserWarning, match="No PIMPLE dict found"):
        result = PressureVelocityAlgorithm.detect_and_create()
    assert result is pimple
