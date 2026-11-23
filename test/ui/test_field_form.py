# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The per-field keywords of a numerics form."""

from __future__ import annotations

import pytest

from neofoam.mcp.registry import resolve_solver
from neofoam.ui.forms import build_forms

_NEON = "incompressibleFluidNeoN"


def _entry(config: str):
    return next(e for e in build_forms(resolve_solver(_NEON)) if e.cls_name == config)


@pytest.mark.parametrize("config", ["PimpleNeoN_fvSolution", "SimpleNeoN_fvSolution"])
def test_solvers_section_groups_blocks_by_field(config):
    schema = _entry(config).form_schema
    section = schema["properties"]["solvers"]

    assert schema["nfFieldForm"] is True
    assert section["nfFieldGroups"] == {"U": ["U", "UFinal"], "p": ["p", "pFinal"]}
    assert section["nfFinalOf"] == {"UFinal": "U", "pFinal": "p"}


def test_backend_note_comes_from_the_model():
    note = _entry("PimpleNeoN_fvSolution").form_schema["nfBackend"]

    assert note["name"] == "Ginkgo"
    assert {"PCG", "GAMG", "DIC"} <= set(note["openfoam"])


@pytest.mark.parametrize(("field", "refused"), [("U", True), ("UFinal", True), ("p", False)])
def test_only_vector_fields_refuse_gmres(field, refused):
    block = _entry("PimpleNeoN_fvSolution").form_schema["properties"]["solvers"]["properties"][
        field
    ]

    assert ("Gmres" in block.get("nfRefusedArms", [])) is refused


def test_config_file_arm_is_labelled_and_drops_the_limits_it_refuses():
    block = _entry("PimpleNeoN_fvSolution").form_schema["properties"]["solvers"]["properties"]["p"]
    arm = next(arm for arm in block["oneOf"] if "configFile" in arm["properties"])

    assert arm["nfArmLabel"] == "Config file"
    assert set(arm["nfSwitchDrops"]) == {"tolerance", "relTol", "maxIter", "minIter"}


@pytest.mark.parametrize("solver", ["incompressibleVoF"])
def test_other_solvers_get_no_per_field_form(solver):
    entries = build_forms(resolve_solver(solver))

    assert not any("nfFieldForm" in entry.form_schema for entry in entries)
