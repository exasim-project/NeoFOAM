# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The per-field keywords of a numerics form and the preview of what Save writes."""

from __future__ import annotations

import pytest

from neofoam.mcp.registry import resolve_solver
from neofoam.ui.file_preview import preview_entries, preview_files
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

    assert preview_entries(entries) == []
    assert not any("nfFieldForm" in entry.form_schema for entry in entries)


def test_preview_leaves_an_unset_option_out_and_reports_a_refused_form():
    entries = [e for e in preview_entries(build_forms(resolve_solver(_NEON))) if "Pimple" in e.key]
    state = {entry.key: dict(entry.defaults) for entry in entries}
    solution = state["dict:PimpleNeoN_fvSolution"]

    text = preview_files(entries, state, {"PimpleNeoN"}, None)["system/fvSolution"]
    assert "krylov_dim" not in text and "FoamFile" in text

    solution["solvers"] = {**solution["solvers"], "U": {"solver": "Gmres"}}
    text = preview_files(entries, state, {"PimpleNeoN"}, None)["system/fvSolution"]
    assert text.startswith("// Save would refuse this form:") and "Gmres" in text
