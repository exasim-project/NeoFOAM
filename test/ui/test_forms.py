# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for the JSONForms form registry (no trame, no JS)."""

from __future__ import annotations

from neofoam.agent.case_fill import build_case_output_model
from neofoam.agent.case_forms import INPUT_KEYS
from neofoam.mcp import tools
from neofoam.mcp.registry import resolve_solver
from neofoam.ui.forms import build_forms


def _solver():
    return resolve_solver("incompressibleFluid")


_MESH_FILES = {"blockMeshDict", "snappyHexMeshDict", "preprocess.yaml"}


def _dict_and_field_names(solver):
    dicts, fields = set(), set()
    for info in tools.list_configs(solver):
        if info.file and info.file.rsplit("/", 1)[-1] in _MESH_FILES:
            continue  # excluded from the wizard (upstream meshing stage)
        (fields if info.file and info.file.startswith("0/") else dicts).add(info.name)
    return dicts, fields


def test_dict_configs_get_one_entry_field_configs_get_two():
    solver = _solver()
    entries = build_forms(solver)
    dicts, fields = _dict_and_field_names(solver)

    dict_entries = [e for e in entries if e.kind == "dict"]
    assert {e.config_name for e in dict_entries} == dicts

    for cfg in fields:
        halves = [e for e in entries if e.config_name == cfg and e.kind != "dict"]
        kinds = sorted(e.kind for e in halves)
        assert kinds == ["field_bc", "field_in"], f"{cfg} missing a half"


def test_field_halves_have_sliced_schemas():
    solver = _solver()
    entries = build_forms(solver)
    for e in entries:
        if e.kind == "field_in":
            assert set(e.schema.get("properties", {})) <= set(INPUT_KEYS)
        elif e.kind == "field_bc":
            assert set(e.schema.get("properties", {})) <= {"boundaryField"}


def test_dict_entry_schema_is_jsonforms_transformed_config_schema():
    from neofoam.ui.forms import jsonforms_schema

    solver = _solver()
    for e in build_forms(solver):
        if e.kind != "dict":
            continue
        schema = tools.config_schema(solver, e.config_name)
        # The entry schema is the config's JSON schema after the JSONForms transform.
        assert e.schema == jsonforms_schema(schema.json_schema)
        assert e.defaults == schema.defaults


def test_jsonforms_transform_unwraps_optional_and_titles_unions():
    from neofoam.io.pydantic_schema import slice_schema
    from neofoam.ui.forms import jsonforms_schema

    solver = _solver()
    # Optional[X] (anyOf[X, null]) collapses to X — turbulence RAS/LES.
    turb = jsonforms_schema(
        tools.config_schema(solver, "turbulence_properties_config").json_schema
    )
    assert "anyOf" not in turb["properties"]["RAS"]
    assert "anyOf" not in turb["properties"]["LES"]
    # Discriminated BC union becomes a oneOf titled by the const `type`.
    bc = jsonforms_schema(
        slice_schema(
            tools.config_schema(solver, "u_field_config").json_schema, ["boundaryField"]
        )
    )
    arms = bc["properties"]["boundaryField"]["additionalProperties"]["oneOf"]
    titles = {a.get("title") for a in arms}
    assert {"fixedValue", "noSlip", "zeroGradient"} <= titles


def test_field_in_dimensions_and_internalfield_collapse_to_text():
    # The fixed 7-element dimension vector and the FieldValue union both render as a
    # single string field, not a growable integer list / ANYOF combinator tabs.
    entries = {e.key: e for e in build_forms(_solver())}
    props = entries["field_in:UFieldConfig"].schema["properties"]

    dims = props["dimensions"]
    assert dims["type"] == "string"
    assert dims["default"] == "[0 1 -1 0 0 0 0]"

    internal = props["internalField"]
    assert internal["type"] == "string"
    assert "anyOf" not in internal and "oneOf" not in internal
    assert internal["default"] == "uniform (0 0 0)"


def test_bc_value_arm_collapses_to_text_but_type_union_stays():
    from neofoam.io.pydantic_schema import slice_schema
    from neofoam.ui.forms import jsonforms_schema

    bc = jsonforms_schema(
        slice_schema(
            tools.config_schema(_solver(), "u_field_config").json_schema,
            ["boundaryField"],
        )
    )
    # The discriminated BC-type union is preserved (a titled oneOf) …
    fixed = next(
        a
        for a in bc["properties"]["boundaryField"]["additionalProperties"]["oneOf"]
        if a.get("title") == "fixedValue"
    )
    fixed_def = bc["$defs"]["FixedValueBC"]
    # … but the fixedValue `value` (scalar|vector|uniform-string) collapses to a string.
    assert fixed_def["properties"]["value"]["type"] == "string"
    assert fixed["$ref"].endswith("FixedValueBC")


def test_allowed_bc_types_lists_titled_arms():
    from neofoam.ui.forms import allowed_bc_types

    entries = {e.key: e for e in build_forms(_solver())}
    u = allowed_bc_types(entries["field_bc:UFieldConfig"])
    p = allowed_bc_types(entries["field_bc:pFieldConfig"])
    assert {"noSlip", "fixedValue", "zeroGradient", "empty"} <= set(u)
    assert "noSlip" in u and "noSlip" not in p  # vector-only arm


def test_seed_boundary_field_roles_values_and_preservation():
    from neofoam.ui.forms import seed_boundary_field

    entries = {e.key: e for e in build_forms(_solver())}
    patches = [
        {"name": "inlet", "role": "inlet"},
        {"name": "outlet", "role": "outlet"},
        {"name": "walls", "role": "wall"},
        {"name": "frontBack", "role": "empty"},
    ]
    u = seed_boundary_field(entries["field_bc:UFieldConfig"], patches, {})[
        "boundaryField"
    ]
    # Role → BC type, clamped to the field's allowed arms.
    assert u["frontBack"] == {"type": "empty"}
    assert u["walls"] == {"type": "noSlip"}
    assert u["outlet"] == {"type": "zeroGradient"}
    # Value-carrying types are seeded with a valid zero payload (matching the arm).
    assert u["inlet"] == {"type": "fixedValue", "value": "uniform (0 0 0)"}
    # A scalar field gets the scalar zero literal.
    p = seed_boundary_field(entries["field_bc:pFieldConfig"], patches, {})[
        "boundaryField"
    ]
    assert p["inlet"] == {"type": "fixedValue", "value": "uniform 0"}
    # Existing entries are preserved (not overwritten).
    keep = {
        "boundaryField": {"inlet": {"type": "fixedValue", "value": "uniform (1 0 0)"}}
    }
    out = seed_boundary_field(entries["field_bc:UFieldConfig"], patches, keep)
    assert out["boundaryField"]["inlet"]["value"] == "uniform (1 0 0)"


def test_patch_bc_schema_pins_named_patch_sections():
    from neofoam.ui.forms import patch_bc_schema

    entries = {e.key: e for e in build_forms(_solver())}
    entry = entries["field_bc:UFieldConfig"]
    schema = patch_bc_schema(entry, ["inlet", "walls"])
    bf = schema["properties"]["boundaryField"]
    assert bf["type"] == "object"
    assert set(bf["properties"]) == {"inlet", "walls"}
    assert bf["properties"]["inlet"]["title"] == "inlet"
    assert "oneOf" in bf["properties"]["inlet"]  # each patch keeps the BC-type union
    assert "$defs" in schema  # arm $refs still resolvable
    # No names → schema returned unchanged.
    assert patch_bc_schema(entry, []) is entry.schema


def test_step_assignment():
    entries = build_forms(_solver())
    for e in entries:
        if e.kind == "field_in":
            assert e.step == "initial"
        elif e.kind == "field_bc":
            assert e.step == "bcs"
        else:
            assert e.step in ("models", "schemes")
    # fv* configs land in schemes.
    schemes = {e.config_name for e in entries if e.step == "schemes"}
    assert "pimple_fv_schemes" in schemes
    assert "pimple_fv_solution" in schemes


def test_owner_model_only_for_optional_configs():
    entries = build_forms(_solver())
    owners = {e.config_name: e.owner_model for e in entries}
    # Boussinesq (optional) configs are owned; core/required configs are not.
    assert owners["boussinesq_config"] == "boussinesq"
    assert owners["t_field_config"] == "boussinesq"
    assert owners["control_dict_config"] is None
    assert owners["transport_properties_config"] is None  # Newtonian is required


def test_config_names_are_valid_save_case_fields():
    solver = _solver()
    valid = set(build_case_output_model(solver=solver).model_fields)
    for e in build_forms(solver):
        assert e.config_name in valid


def test_state_keys_unique():
    entries = build_forms(_solver())
    keys = [e.state_key for e in entries]
    assert len(keys) == len(set(keys))
