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


@pytest.mark.parametrize("solver", ["incompressibleFluid", _NEON])
@pytest.mark.parametrize(
    ("model", "rows"),
    [
        ("kEpsilon", {"k", "epsilon"}),
        ("realizableKE", {"k", "epsilon"}),
        ("kOmegaSST", {"k", "omega"}),
        ("SpalartAllmaras", {"nuTilda"}),
    ],
)
@pytest.mark.parametrize("file", ["fvSchemes", "fvSolution"])
def test_turbulence_model_owns_the_rows_of_its_fields(solver, model, rows, file):
    entry = next(e for e in build_forms(resolve_solver(solver)) if e.cls_name == f"{model}_{file}")
    sections = entry.form_schema["properties"].values()
    declared = [item for s in sections for item in s.get("properties", {}).values()]

    assert (entry.owner_model, {item["nfField"] for item in declared}) == (model, rows)


@pytest.mark.parametrize("solver", ["incompressibleFluid", _NEON])
def test_every_entry_of_a_field_section_has_a_row(solver):
    forms = [e for e in build_forms(resolve_solver(solver)) if e.field_schema]
    sections = [s for e in forms for s in e.field_schema["properties"].values()]
    field_sections = [
        s for s in sections if any("nfField" in p for p in s.get("properties", {}).values())
    ]

    assert [
        key
        for section in field_sections
        for key, entry in section["properties"].items()
        if "nfField" not in entry
    ] == []


def test_boussinesq_declares_the_rows_of_its_fields():
    forms = {e.cls_name: e for e in build_forms(resolve_solver("incompressibleFluid"))}
    solvers = forms["boussinesq_fvSolution"].form_schema["properties"]["solvers"]

    assert solvers["nfFieldGroups"] == {"T": ["T", "TFinal"], "p_rgh": ["p_rgh", "p_rghFinal"]}


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
