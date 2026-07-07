# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the case form-wiring helpers (``neofoam.agent.case_forms``).

Exercises the field input/BC split, the save-time merge, and the small
categorisation helpers — all against the real ``incompressibleFluid`` config
classes (no LLM, no marimo). BC-value → OpenFOAM-literal mapping now lives in the
``FieldValue`` type (see ``test/fields/test_value_types.py``), not here.
"""

from __future__ import annotations

from neofoam.agent.case_forms import (
    INPUT_KEYS,
    field_name,
    is_scheme_config,
    merge_field_config,
    split_field_dump,
)
from neofoam.framework.solver.configurations import configurations
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid


def _cfgs():
    return configurations(incompressibleFluid)


def test_is_scheme_config_and_field_name() -> None:
    cfgs = _cfgs()
    assert is_scheme_config(cfgs["Pimple_fvSchemes"]) is True
    assert is_scheme_config(cfgs["Pimple_fvSolution"]) is True
    assert is_scheme_config(cfgs["ControlDictConfig"]) is False
    assert is_scheme_config(cfgs["UFieldConfig"]) is False
    assert field_name(cfgs["UFieldConfig"]) == "U"
    assert field_name(cfgs["pFieldConfig"]) == "p"


def test_split_field_dump() -> None:
    dump = {
        "FoamFile": {"object": "U"},
        "dimensions": [0, 1, -1, 0, 0, 0, 0],
        "internalField": "uniform (0 0 0)",
        "boundaryField": {"walls": {"type": "noSlip"}},
    }
    input_half, bc_half = split_field_dump(dump)
    assert set(input_half) <= set(INPUT_KEYS)
    assert input_half["internalField"] == "uniform (0 0 0)"
    assert "boundaryField" not in input_half
    assert bc_half == {"boundaryField": {"walls": {"type": "noSlip"}}}


def test_split_field_dump_missing_boundaryfield() -> None:
    _, bc_half = split_field_dump({"internalField": "uniform 0"})
    assert bc_half == {"boundaryField": {}}


def test_merge_field_config_merges_halves() -> None:
    U = _cfgs()["UFieldConfig"]
    inst = merge_field_config(
        U,
        {"internalField": "uniform (0 0 0)"},
        {"boundaryField": {"movingWall": {"type": "fixedValue", "value": [1, 0, 0]}}},
    )
    # Raw value kept on the instance (forms see this); OpenFOAM literal is emitted
    # only under the writer's serialization context.
    dump = inst.model_dump(by_alias=True, exclude_none=True)
    assert dump["boundaryField"]["movingWall"]["value"] == [1.0, 0.0, 0.0]
    of = inst.model_dump(
        by_alias=True, exclude_none=True, context={"format": "openfoam"}
    )
    assert of["boundaryField"]["movingWall"]["value"] == "uniform (1.0 0.0 0.0)"


def test_merge_field_config_unsubmitted_half_falls_back_to_default() -> None:
    U = _cfgs()["UFieldConfig"]
    # No input half submitted → internalField comes from the config default.
    inst = merge_field_config(U, None, {"boundaryField": {"walls": {"type": "noSlip"}}})
    dump = inst.model_dump(by_alias=True, exclude_none=True)
    assert "internalField" in dump  # default supplied, not vanished
    assert dump["boundaryField"]["walls"] == {"type": "noSlip"}
