# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``coupled_entries`` on the field files of ``cases/coupledEntries``."""

from pathlib import Path

import pytest

from neofoam.solver.regions import coupled_entries

CASES = Path(__file__).parent / "cases" / "coupledEntries"
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
