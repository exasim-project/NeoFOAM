# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for the JSONForms form registry (no trame, no JS)."""

from __future__ import annotations

from neofoam.agent.case_fill import build_case_output_model
from neofoam.agent.case_forms import INPUT_KEYS
from neofoam.mcp import tools
from neofoam.mcp.registry import list_solver_names, resolve_solver
from neofoam.ui.forms import (
    build_field_forms,
    build_forms,
    build_mesh_forms,
    exclusive_model_families,
    schema_key,
    uischema_key,
)

# Mirrors forms._NON_WIZARD_FILES on purpose: an independent restatement of what the
# wizard drops, so this test does not track the implementation tautologically.
_NON_WIZARD_FILES = {
    "blockMeshDict",
    "snappyHexMeshDict",
    "preprocess.yaml",
    "setFields.yaml",
    "postProcess.yaml",
}


def _dict_and_field_names(solver):
    dicts, fields = set(), set()
    for info in tools.list_configs(solver):
        if info.file and info.file.rsplit("/", 1)[-1] in _NON_WIZARD_FILES:
            continue  # excluded from the wizard (meshing / pre- / post-processing stage)
        (fields if info.file and info.file.startswith("0/") else dicts).add(info.name)
    return dicts, fields


def test_dict_configs_get_one_entry_field_configs_get_two(solver):
    entries = build_forms(solver)
    dicts, fields = _dict_and_field_names(solver)

    dict_entries = [e for e in entries if e.kind == "dict"]
    assert {e.config_name for e in dict_entries} == dicts

    for cfg in fields:
        halves = [e for e in entries if e.config_name == cfg and e.kind != "dict"]
        kinds = sorted(e.kind for e in halves)
        assert kinds == ["field_bc", "field_in"], f"{cfg} missing a half"


def test_field_halves_have_sliced_schemas(solver):
    entries = build_forms(solver)
    for e in entries:
        if e.kind == "field_in":
            assert set(e.schema.get("properties", {})) <= set(INPUT_KEYS)
        elif e.kind == "field_bc":
            assert set(e.schema.get("properties", {})) <= {"boundaryField"}


def test_dict_entry_schema_is_jsonforms_transformed_config_schema(solver):
    from neofoam.ui.form_schema import jsonforms_schema  # noqa: PLC0415

    for e in build_forms(solver):
        if e.kind != "dict":
            continue
        schema = tools.config_schema(solver, e.config_name)
        # The entry schema is the config's JSON schema after the JSONForms transform.
        assert e.schema == jsonforms_schema(schema.json_schema)
        assert e.defaults == schema.defaults


def test_step_assignment(solver):
    entries = build_forms(solver)
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


def test_owner_model_only_for_optional_configs(solver):
    entries = build_forms(solver)
    owners = {e.config_name: e.owner_model for e in entries}
    # Boussinesq (optional) configs are owned; core/required configs are not.
    assert owners["boussinesq_config"] == "boussinesq"
    assert owners["t_field_config"] == "boussinesq"
    assert owners["control_dict_config"] is None
    assert owners["transport_properties_config"] is None  # Newtonian is required


def test_exclusive_model_families_are_the_multi_member_required_ones(solver):
    families = exclusive_model_families(solver)
    # Pick ONE: the pressure-velocity algorithm and the turbulence model.
    assert families["PressureVelocityAlgorithm"] == ["Pimple", "Simple"]
    assert families["momentumTransportModel"][:2] == ["kEpsilon", "kOmegaSST"]
    # viscosityModel is required with a single member (nothing to choose), and
    # incompressibleFluidModel is optional (independent toggles) — neither is a choice.
    assert set(families) == {"PressureVelocityAlgorithm", "momentumTransportModel"}


def test_family_members_own_their_own_dicts_but_not_the_shared_configs(solver):
    owners = {(e.config_name, e.kind): e.owner_model for e in build_forms(solver)}
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


def test_config_names_are_valid_save_case_fields(solver):
    valid = set(build_case_output_model(solver=solver).model_fields)
    for e in build_forms(solver):
        assert e.config_name in valid


def test_state_keys_unique(solver):
    entries = build_forms(solver)
    keys = [e.state_key for e in entries]
    assert len(keys) == len(set(keys))


def test_build_mesh_forms_surfaces_sweepable_mesh_dicts_only(solver):
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


def test_build_field_forms_are_whole_field_dict_entries(solver):
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
    for name in list_solver_names():
        for entry in build_forms(resolve_solver(name)):
            for var in (entry.state_key, schema_key(entry), uischema_key(entry)):
                assert var.isidentifier(), f"{name}: {var!r}"


def test_neon_panel_titles_keep_neon_whole():
    titles = [e.title for e in build_forms(resolve_solver("incompressibleFluidNeoN"))]

    assert "controlDict · NeoN Control" in titles
    assert not [t for t in titles if "Neo N" in t]
