# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for the JSONForms form registry (no trame, no JS)."""

from __future__ import annotations

import pytest

from neofoam.agent.case_fill import build_case_output_model
from neofoam.agent.case_forms import INPUT_KEYS
from neofoam.mcp import tools
from neofoam.mcp.registry import list_solver_names, resolve_solver
from neofoam.ui.forms import (
    build_field_forms,
    build_forms,
    build_mesh_forms,
    exclusive_model_families,
)


def _solver():
    return resolve_solver("incompressibleFluid")


def _titles(node, path=""):
    """Every ``(json-pointer, title)`` in a schema, however deeply nested."""
    if isinstance(node, dict):
        title = node.get("title")
        if isinstance(title, str):
            yield path, title
        for key, value in node.items():
            yield from _titles(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _titles(value, f"{path}/{index}")


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
    from neofoam.ui.forms import jsonforms_schema  # noqa: PLC0415

    solver = _solver()
    for e in build_forms(solver):
        if e.kind != "dict":
            continue
        schema = tools.config_schema(solver, e.config_name)
        # The entry schema is the config's JSON schema after the JSONForms transform.
        assert e.schema == jsonforms_schema(schema.json_schema)
        assert e.defaults == schema.defaults


def test_jsonforms_transform_unwraps_optional_and_titles_unions():
    from neofoam.io.pydantic_schema import slice_schema  # noqa: PLC0415
    from neofoam.ui.forms import jsonforms_schema  # noqa: PLC0415

    solver = _solver()
    # Optional[X] (anyOf[X, null]) collapses to X — turbulence RAS/LES.
    turb = jsonforms_schema(tools.config_schema(solver, "turbulence_properties_config").json_schema)
    assert "anyOf" not in turb["properties"]["RAS"]
    assert "anyOf" not in turb["properties"]["LES"]
    # Discriminated BC union becomes a oneOf titled by the const `type`.
    bc = jsonforms_schema(
        slice_schema(tools.config_schema(solver, "u_field_config").json_schema, ["boundaryField"])
    )
    arms = bc["properties"]["boundaryField"]["additionalProperties"]["oneOf"]
    titles = {a.get("title") for a in arms}
    assert {"fixedValue", "noSlip", "zeroGradient"} <= titles


def test_openfoam_dict_keys_are_titled_verbatim():
    # Keys of a synthesized fvSchemes/fvSolution section are OpenFOAM entries the
    # case author wrote, not an API surface: humanizing them destroys information a
    # user cannot recover (p_rghFinal read as "P Rghfinal") and no longer names
    # anything in the written case file.
    entries = {e.key: e for e in build_forms(_solver())}
    # The buoyant pressure solvers sit on the Boussinesq model's own slice.
    solvers = {
        **entries["dict:Pimple_fvSolution"].schema["properties"]["solvers"]["properties"],
        **entries["dict:boussinesq_fvSolution"].schema["properties"]["solvers"]["properties"],
    }
    assert [solvers[k]["title"] for k in ("p", "pFinal", "p_rgh", "p_rghFinal")] == [
        "p",
        "pFinal",
        "p_rgh",
        "p_rghFinal",
    ]
    div = entries["dict:Pimple_fvSchemes"].schema["properties"]["divSchemes"]["properties"]
    assert div["div(phi,U)"]["title"] == "div(phi,U)"
    # A dotted phase field survives too ("Alpha.water" before).
    vof = {e.key: e for e in build_forms(resolve_solver("incompressibleVoF"))}
    alpha = vof["dict:MULES_fvSolution"].schema["properties"]["solvers"]["properties"]
    assert alpha["alpha.water"]["title"] == "alpha.water"


def test_section_headings_drop_the_synthesized_class_name():
    # fv_configs._rebuild_sections names each section model "_<section>"; that
    # leading underscore must not reach the panel ("_ddtSchemes", "_PIMPLE").
    entries = {e.key: e for e in build_forms(_solver())}
    schemes = entries["dict:Pimple_fvSchemes"].schema["properties"]
    assert schemes["ddtSchemes"]["title"] == "ddtSchemes"
    solution = entries["dict:Pimple_fvSolution"].schema["properties"]
    assert solution["PIMPLE"]["title"] == "PIMPLE"
    assert solution["solvers"]["title"] == "solvers"
    # No private class name leaks anywhere, in any solver.
    for name in list_solver_names():
        for entry in build_forms(resolve_solver(name)):
            for path, title in _titles(entry.schema):
                assert not title.startswith("_"), f"{name}/{entry.key}{path}: {title}"


def test_sections_of_like_entries_keep_the_add_a_key_row():
    # JSONForms offers an "add a key" row for `additionalProperties`. A section is a
    # collection — divSchemes gains a div(phi,k), solvers a new solver block — so the
    # row stays there, single-entry sections included.
    entries = {e.key: e for e in build_forms(_solver())}
    schemes = entries["dict:Pimple_fvSchemes"].schema["properties"]
    assert schemes["divSchemes"]["additionalProperties"] is True
    assert schemes["ddtSchemes"]["additionalProperties"] is True
    solvers = entries["dict:Pimple_fvSolution"].schema["properties"]["solvers"]
    assert solvers["additionalProperties"] is True


def test_fixed_shape_cards_offer_no_add_a_key_row():
    # `additionalProperties: true` on a card of differently-shaped entries can only
    # add an untyped value — noise. Dropping the keyword hides the row and still
    # validates the same data (absent means "allowed").
    entries = {e.key: e for e in build_forms(_solver())}
    solution = entries["dict:Pimple_fvSolution"].schema
    assert "additionalProperties" not in solution
    assert "additionalProperties" not in solution["properties"]["PIMPLE"]
    assert "additionalProperties" not in entries["dict:Pimple_fvSchemes"].schema
    vof = {e.key: e for e in build_forms(resolve_solver("incompressibleVoF"))}
    water = vof["dict:TransportPropertiesConfig"].schema["properties"]["water"]
    assert "additionalProperties" not in water


def test_open_dicts_without_known_keys_keep_the_add_a_key_row():
    # A solver block (and MRFProperties / fvOptions) declares no keys at all: every
    # entry is an additional property, so the row is the only way to edit it.
    entries = {e.key: e for e in build_forms(_solver())}
    block = entries["dict:Pimple_fvSolution"].schema["properties"]["solvers"]["properties"]["p"]
    assert block["additionalProperties"] is True
    assert entries["dict:FvOptionsConfig"].schema["additionalProperties"] is True


def test_nested_model_is_titled_by_its_key_not_its_class_name():
    # A `water: PhaseTransport` property inherits the *class* name as its title via
    # the $ref, so both phase cards read "PhaseTransport"; the key is what tells
    # them apart.
    vof = {e.key: e for e in build_forms(resolve_solver("incompressibleVoF"))}
    transport = vof["dict:TransportPropertiesConfig"].schema["properties"]
    assert [transport[k]["title"] for k in ("water", "air")] == ["Water", "Air"]


@pytest.mark.parametrize("solver_name", ["incompressibleFluid", "incompressibleVoF"])
def test_foamfile_header_stays_in_the_data_but_is_not_rendered(solver_name):
    # version / format / class / object are boilerplate the writer needs and the
    # user never edits.
    entries = {e.key: e for e in build_forms(resolve_solver(solver_name))}
    gravity = entries["dict:GravityConfig"]
    assert "FoamFile" not in gravity.schema["properties"]
    assert gravity.defaults["FoamFile"]["object"] == "g"


def test_prose_property_keys_are_still_humanized():
    # Hand-written config models keep their human labels — only OpenFOAM blocks
    # are shown verbatim.
    entries = {e.key: e for e in build_forms(_solver())}
    control = entries["dict:ControlDictConfig"].schema["properties"]
    assert control["writeControl"]["title"] == "Write Control"
    assert control["deltaT"]["title"] == "Delta T"
    pimple = entries["dict:PimpleAlgorithmConfig"].schema["properties"]
    assert pimple["momentumPredictor"]["title"] == "Momentum Predictor"


def test_inline_refs_makes_every_schema_self_contained():
    # JSONForms' combinator renderers ajv.compile() every oneOf/anyOf arm
    # STANDALONE; an arm containing a $ref throws, is never cached and
    # recompiles on every reactive re-evaluation — the big scheme schemas
    # (nested discriminated unions) hard-froze the page that way.
    import json  # noqa: PLC0415

    for name in list_solver_names():
        for e in build_forms(resolve_solver(name)):
            text = json.dumps(e.schema)
            assert "$ref" not in text, f"{name}/{e.key} still contains a $ref"
            assert "$defs" not in e.schema, f"{name}/{e.key} still carries $defs"
            assert "discriminator" not in text, f"{name}/{e.key} keeps a dangling discriminator"


def test_inline_refs_bounds_a_self_referential_union():
    # cellLimited nests a GradScheme, which includes cellLimited again. The cycle
    # cannot be inlined, so the inner scheme offers the non-recursive arms only.
    entries = {e.key: e for e in build_forms(_solver())}
    grad = entries["dict:Pimple_fvSchemes"].schema["properties"]["gradSchemes"]["properties"]
    arms = {a["title"]: a for a in grad["grad(U)"]["oneOf"]}
    assert set(arms) == {"Gauss", "pointCellsLeastSquares", "cellLimited"}
    inner = arms["cellLimited"]["properties"]["inner_scheme"]["oneOf"]
    assert [a["title"] for a in inner] == ["Gauss", "pointCellsLeastSquares"]


def _discriminators(node):
    """Every ``type`` property schema that pins a ``const``, however deeply nested."""
    if isinstance(node, dict):
        prop = node.get("properties", {}).get("type")
        if isinstance(prop, dict) and "const" in prop:
            yield prop
        for value in node.values():
            yield from _discriminators(value)
    elif isinstance(node, list):
        for value in node:
            yield from _discriminators(value)


def test_union_arm_discriminator_is_kept_in_the_data_but_not_rendered():
    # The arm selector already names the type; a second "Type" dropdown holding that one
    # value under every (nested) scheme made the Numerics step 7000-12000 px tall.
    # JSONForms generates a control only for a property it can derive a JSON type for
    # (`type`/`enum`/`properties`/`items`), and fills a newly selected arm's data from
    # `default` — so exactly `const` + `default` hides the control and keeps the value.
    for name in list_solver_names():
        for entry in build_forms(resolve_solver(name)):
            for prop in _discriminators(entry.schema):
                assert prop == {"const": prop["const"], "default": prop["const"]}, (
                    f"{name}/{entry.key}: {prop}"
                )
    div = {e.key: e for e in build_forms(_solver())}["dict:Pimple_fvSchemes"].schema
    arms = div["properties"]["divSchemes"]["properties"]["div(phi,U)"]["oneOf"]
    assert [(a["title"], list(a["properties"])) for a in arms] == [
        ("none", ["type"]),
        ("Gauss", ["type", "interpolation"]),
        ("bounded", ["type", "interpolation"]),
    ]


def test_inline_refs_merges_siblings_and_keeps_cycles():
    from neofoam.ui.forms import inline_refs  # noqa: PLC0415

    schema = {
        "$defs": {
            "Leaf": {"type": "object", "title": "leaf-title"},
            "Loop": {"properties": {"next": {"$ref": "#/$defs/Loop"}}},
        },
        "properties": {
            "a": {"$ref": "#/$defs/Leaf", "title": "arm-title"},
            "b": {"$ref": "#/$defs/Loop"},
        },
    }
    out = inline_refs(schema)
    # Sibling keys override the resolved definition.
    assert out["properties"]["a"] == {"type": "object", "title": "arm-title"}
    # The cyclic ref survives, so its definitions are kept.
    assert out["properties"]["b"]["properties"]["next"]["$ref"] == "#/$defs/Loop"
    assert "Loop" in out["$defs"]


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
    from neofoam.io.pydantic_schema import slice_schema  # noqa: PLC0415
    from neofoam.ui.forms import jsonforms_schema  # noqa: PLC0415

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
    # … every arm is self-contained (inline_refs: no $ref/$defs — JSONForms'
    # combinator renderer must be able to hand each arm to AJV standalone) …
    assert "$ref" not in fixed and "$defs" not in bc
    # … and the fixedValue `value` (scalar|vector|uniform-string) collapses to a string.
    assert fixed["properties"]["value"]["type"] == "string"


def test_allowed_bc_types_lists_titled_arms():
    from neofoam.ui.forms import allowed_bc_types  # noqa: PLC0415

    entries = {e.key: e for e in build_forms(_solver())}
    u = allowed_bc_types(entries["field_bc:UFieldConfig"])
    p = allowed_bc_types(entries["field_bc:pFieldConfig"])
    assert {"noSlip", "fixedValue", "zeroGradient", "empty"} <= set(u)
    assert "noSlip" in u and "noSlip" not in p  # vector-only arm


def test_seed_boundary_field_roles_values_and_preservation():
    from neofoam.ui.forms import seed_boundary_field  # noqa: PLC0415

    entries = {e.key: e for e in build_forms(_solver())}
    patches = [
        {"name": "inlet", "role": "inlet"},
        {"name": "outlet", "role": "outlet"},
        {"name": "walls", "role": "wall"},
        {"name": "frontBack", "role": "empty"},
    ]
    u = seed_boundary_field(entries["field_bc:UFieldConfig"], patches, {})["boundaryField"]
    # Role → BC type, clamped to the field's allowed arms. Role `empty` seeds a
    # symmetry BC — the snappy mesh realises empty patches as symmetry patches.
    assert u["frontBack"] == {"type": "symmetry"}
    assert u["walls"] == {"type": "noSlip"}
    assert u["outlet"] == {"type": "zeroGradient"}
    # Value-carrying types are seeded with a valid zero payload (matching the arm).
    assert u["inlet"] == {"type": "fixedValue", "value": "uniform (0 0 0)"}
    # A scalar field gets the scalar zero literal.
    p = seed_boundary_field(entries["field_bc:pFieldConfig"], patches, {})["boundaryField"]
    assert p["inlet"] == {"type": "fixedValue", "value": "uniform 0"}
    # Existing entries are preserved (not overwritten).
    keep = {"boundaryField": {"inlet": {"type": "fixedValue", "value": "uniform (1 0 0)"}}}
    out = seed_boundary_field(entries["field_bc:UFieldConfig"], patches, keep)
    assert out["boundaryField"]["inlet"]["value"] == "uniform (1 0 0)"


def test_patch_bc_schema_pins_named_patch_sections():
    from neofoam.ui.forms import patch_bc_schema  # noqa: PLC0415

    entries = {e.key: e for e in build_forms(_solver())}
    entry = entries["field_bc:UFieldConfig"]
    schema = patch_bc_schema(entry, ["inlet", "walls"])
    bf = schema["properties"]["boundaryField"]
    assert bf["type"] == "object"
    assert set(bf["properties"]) == {"inlet", "walls"}
    assert bf["properties"]["inlet"]["title"] == "inlet"
    assert "oneOf" in bf["properties"]["inlet"]  # each patch keeps the BC-type union
    assert "$defs" not in schema  # arms are inlined — nothing left to resolve
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


def test_exclusive_model_families_are_the_multi_member_required_ones():
    families = exclusive_model_families(_solver())
    # Pick ONE: the pressure-velocity algorithm and the turbulence model.
    assert families["PressureVelocityAlgorithm"] == ["Pimple", "Simple"]
    assert families["momentumTransportModel"][:2] == ["kEpsilon", "kOmegaSST"]
    # viscosityModel is required with a single member (nothing to choose), and
    # incompressibleFluidModel is optional (independent toggles) — neither is a choice.
    assert set(families) == {"PressureVelocityAlgorithm", "momentumTransportModel"}


def test_family_members_own_their_own_dicts_but_not_the_shared_configs():
    owners = {(e.config_name, e.kind): e.owner_model for e in build_forms(_solver())}
    # Each alternative gates its own dictionaries — Pimple and Simple must never show
    # (or write) their contradictory fvSchemes/fvSolution slices at the same time.
    assert owners[("pimple_fv_schemes", "dict")] == "Pimple"
    assert owners[("simple_fv_schemes", "dict")] == "Simple"
    assert owners[("k_epsilon_coeffs", "dict")] == "kEpsilon"
    # Configs the whole family shares stay ungated: turbulenceProperties is declared by
    # every turbulence model, and 0/U by both algorithms.
    assert owners[("turbulence_properties_config", "dict")] is None
    assert owners[("u_field_config", "field_in")] is None
    assert owners[("u_field_config", "field_bc")] is None
    # Newtonian is required with no alternatives — always on.
    assert owners[("transport_properties_config", "dict")] is None


def test_turbulence_properties_uischema_gates_ras_and_les_on_simulation_type():
    entries = {e.config_name: e for e in build_forms(_solver())}
    uischema = entries["turbulence_properties_config"].uischema
    assert uischema is not None
    rules = {
        e["scope"].rsplit("/", 1)[-1]: e.get("rule")
        for e in uischema["elements"]
        if isinstance(e, dict)
    }
    # simulationType picks; RAS/LES render only when it names them.
    assert rules["simulationType"] is None
    for block in ("RAS", "LES"):
        assert rules[block] == {
            "effect": "SHOW",
            "condition": {
                "scope": "#/properties/simulationType",
                "schema": {"const": block},
            },
        }
    # No other form needs a hand-written layout (JSONForms auto-generates it).
    assert [e.config_name for e in build_forms(_solver()) if e.uischema is not None] == [
        "turbulence_properties_config"
    ]


def test_config_names_are_valid_save_case_fields():
    solver = _solver()
    valid = set(build_case_output_model(solver=solver).model_fields)
    for e in build_forms(solver):
        assert e.config_name in valid


def test_state_keys_unique():
    entries = build_forms(_solver())
    keys = [e.state_key for e in entries]
    assert len(keys) == len(set(keys))


def test_build_mesh_forms_surfaces_sweepable_mesh_dicts_only():
    solver = _solver()
    mesh = {e.config_name: e for e in build_mesh_forms(solver)}
    # blockMesh + snappy are surfaced (for the Parameters step's mesh dimension);
    # preprocess.yaml (a YAML config, not a per-config sweep target) is not.
    assert set(mesh) == {"block_mesh_dict_config", "snappy_hex_mesh_dict_config"}
    for entry in mesh.values():
        assert entry.kind == "dict"
        assert entry.step == "mesh"  # never a wizard step
        assert entry.owner_model is None
        assert entry.schema.get("properties")
    # And they are kept OUT of the physics wizard forms.
    wizard = {e.config_name for e in build_forms(solver)}
    assert wizard.isdisjoint(mesh)


def test_build_field_forms_are_whole_field_dict_entries():
    solver = _solver()
    fields = {e.config_name: e for e in build_field_forms(solver)}
    # One whole-field entry per 0/<field> config.
    assert "u_field_config" in fields
    for entry in fields.values():
        assert entry.kind == "dict"  # whole config, not a split half
        assert entry.step == "fields"
        props = entry.schema.get("properties", {})
        # Carries BOTH halves (input + boundaryField), unlike the wizard's split.
        assert "boundaryField" in props
    # T / p_rgh fields are owned by the optional Boussinesq model (gated).
    assert fields["t_field_config"].owner_model == "boussinesq"
    assert fields["u_field_config"].owner_model is None


def test_state_var_names_are_js_identifiers():
    # trame evaluates state var names as Vue expressions: `form_alpha.water…` reads as
    # a member access on an undefined `form_alpha` and the panel renders empty.
    from neofoam.ui.app import _schema_key, _uischema_key  # noqa: PLC0415

    for name in list_solver_names():
        for entry in build_forms(resolve_solver(name)):
            for var in (entry.state_key, _schema_key(entry), _uischema_key(entry)):
                assert var.isidentifier(), f"{name}: {var!r}"


@pytest.mark.parametrize(
    ("key", "label"),
    [
        ("deltaT", "Delta T"),
        ("writeControl", "Write Control"),
        # "NeoN" is a name, not two camelCase words.
        ("PressureVelocityAlgorithmNeoN", "Pressure Velocity Algorithm NeoN"),
        ("NeoNControl", "NeoN Control"),
    ],
)
def test_humanize_splits_words_but_keeps_neon_whole(key, label):
    from neofoam.ui.forms import humanize  # noqa: PLC0415

    assert humanize(key) == label


def test_neon_panel_titles_keep_neon_whole():
    titles = [e.title for e in build_forms(resolve_solver("incompressibleFluidNeoN"))]

    assert "controlDict · NeoN Control" in titles
    assert not [t for t in titles if "Neo N" in t]
