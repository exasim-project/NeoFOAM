# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``neofoam.solver.regions`` on the files of ``cases/coupledEntries``, ``cases/regionSolvers``."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from neofoam.solver.regions import ControlDictConfig, coupled_entries

CASES = Path(__file__).parent / "cases" / "coupledEntries"
REGION_SOLVERS = Path(__file__).parent / "cases" / "regionSolvers"
PATCH_TYPES = {"coupledTemperature": "calculated", "coupledHeatFlux": "calculated"}


def test_coupled_entries_maps_each_coupled_patch_to_its_condition() -> None:
    assert coupled_entries(CASES / "coupled", PATCH_TYPES) == {"interface": "coupledTemperature"}


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknownCondition", "patch interface is coupled as 'coupledPressure'; known: "),
        ("wrongType", "has type fixedValue; a coupledTemperature patch is 'calculated'"),
    ],
)
def test_coupled_entries_rejects_an_unknown_condition_or_type(case: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        coupled_entries(CASES / case, PATCH_TYPES)


def test_region_solvers_take_a_plain_name_as_one_process() -> None:
    config = ControlDictConfig.load(case_dir=REGION_SOLVERS / "entries")

    assert config.solvers() == {"left": "a", "right": "b"}
    assert config.processes() == {"left": 1, "right": 3}


def test_region_solvers_default_a_region_to_one_process() -> None:
    config = ControlDictConfig.load(case_dir=REGION_SOLVERS / "default_processes")

    assert config.processes() == {"left": 1}


@pytest.mark.parametrize("name", ["zero_processes", "unknown_entry", "no_solver"])
def test_region_solvers_reject_an_invalid_entry(name: str) -> None:
    with pytest.raises(ValidationError):
        ControlDictConfig.load(case_dir=REGION_SOLVERS / name)
