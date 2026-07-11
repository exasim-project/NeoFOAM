# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end ``configurations(incompressibleFluid).fields`` checks.

These tests assert that the field declarations on the in-tree ``pimple``
and ``boussinesq`` models surface through ``configurations(...)`` —
exactly the integration the agent layer relies on. Round-tripping the
fields against the staged hotRoom fixture exercises the path
``Model.field`` → ``schema_for`` → ``load_fields`` end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from neofoam.fields.loader import load_fields, save_fields
from neofoam.framework.solver.configurations import configurations
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid


def test_pimple_fields_surface_through_configurations() -> None:
    field_names = {
        cls.io_config.file for cls in configurations(incompressibleFluid).fields
    }
    # ``U`` / ``p`` are pimple's; ``p_rgh`` / ``T`` / ``alphat`` are
    # boussinesq's. The presence of all five confirms ``model_specs``
    # walks both core and optional families.
    expected = {"0/U", "0/p", "0/p_rgh", "0/T", "0/alphat"}
    assert expected <= field_names


def test_load_fields_round_trip_against_hot_room(staged_hot_room: Path) -> None:
    case = staged_hot_room

    before = load_fields(case, solver=incompressibleFluid)
    save_fields(before, case)
    after = load_fields(case, solver=incompressibleFluid)

    # The fields the in-tree models declare must all load against the
    # hotRoom fixture, and round-trip equality is the load/save
    # contract. (Fields the fixture carries but the solver does *not*
    # declare — k, epsilon, nut — are simply not in ``before``.)
    expected_declared = {"U", "p", "p_rgh", "T", "alphat"}
    assert expected_declared <= set(before)
    for name in expected_declared:
        assert before[name].model_dump() == after[name].model_dump(), name


def test_dict_configs_still_present_alongside_fields() -> None:
    """Step 8 must not have demoted any pre-existing dict configs."""
    cfgs = configurations(incompressibleFluid)
    dict_files = {cls.io_config.file for cls in cfgs.dicts if cls.io_config}
    # A spot check on dict configs that pre-date this work; their
    # presence here means ``configurations(...)`` still surfaces them.
    assert "system/controlDict" in dict_files
    assert "constant/transportProperties" in dict_files
