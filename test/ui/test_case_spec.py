# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for the state → case_spec aggregation core."""

from __future__ import annotations

from neofoam.agent.case_fill import build_case_output_model
from neofoam.framework.solver.configurations import configurations
from neofoam.mcp import tools
from neofoam.mcp.registry import resolve_solver
from neofoam.ui.case_spec import (
    configs_to_form_state,
    models_filled_by,
    owned_entry_keys_by_model,
    state_to_case_spec,
)
from neofoam.ui.forms import build_forms


def _solver():
    return resolve_solver("incompressibleFluid")


def _state_from_defaults(entries, only=None):
    """A form_state map seeded from each entry's defaults (or empty unless in `only`)."""
    return {
        e.key: (dict(e.defaults) if (only is None or e.config_name in only) else {})
        for e in entries
    }


def test_aggregate_validates_and_round_trips(tmp_path):
    solver = _solver()
    entries = build_forms(solver)
    state = _state_from_defaults(entries, only={"transport_properties_config"})

    spec = state_to_case_spec(entries, state, selected_models=set())
    assert set(spec) == {"transport_properties_config"}

    # Structurally valid for the aggregate model, and writes via save_case.
    build_case_output_model(solver=solver)(**spec)
    result = tools.save_case(solver, spec, str(tmp_path))
    assert (tmp_path / "constant" / "transportProperties").is_file()
    assert any("transportProperties" in w for w in result.written)


def test_unselected_optional_model_is_skipped():
    solver = _solver()
    entries = build_forms(solver)
    # Fill Boussinesq's data, but do NOT select the model.
    state = _state_from_defaults(entries, only={"boussinesq_config", "gravity_config"})

    spec = state_to_case_spec(entries, state, selected_models=set())
    assert "boussinesq_config" not in spec
    assert "gravity_config" not in spec

    spec_on = state_to_case_spec(entries, state, selected_models={"boussinesq"})
    assert "boussinesq_config" in spec_on


def test_empty_forms_are_omitted():
    solver = _solver()
    entries = build_forms(solver)
    spec = state_to_case_spec(entries, _state_from_defaults(entries, only=set()), set())
    assert spec == {}


def test_field_merges_from_one_half():
    solver = _solver()
    entries = build_forms(solver)
    # Provide only the input half of U; BC half empty → merge falls back to defaults.
    state = {e.key: {} for e in entries}
    for e in entries:
        if e.config_name == "u_field_config" and e.kind == "field_in":
            state[e.key] = dict(e.defaults)

    spec = state_to_case_spec(entries, state, selected_models=set())
    assert "u_field_config" in spec
    assert isinstance(spec["u_field_config"], dict)


def test_owned_entry_keys_by_model():
    entries = build_forms(_solver())
    owned = owned_entry_keys_by_model(entries)
    assert "boussinesq" in owned
    assert any(k.startswith("dict:BoussinesqConfig") for k in owned["boussinesq"])
    # Required models own nothing here (their configs have owner_model=None).
    assert "Newtonian" not in owned


def test_configs_to_form_state_maps_dict_and_field_halves():
    solver = _solver()
    entries = build_forms(solver)
    cfgs = configurations(solver)
    transport = cfgs["TransportPropertiesConfig"].model_construct()
    u_field = cfgs["UFieldConfig"].model_construct()

    fs = configs_to_form_state(entries, [transport, u_field])
    assert "dict:TransportPropertiesConfig" in fs  # dict config → one key
    assert "field_in:UFieldConfig" in fs  # field config → two half keys
    assert "field_bc:UFieldConfig" in fs


def test_configs_to_form_state_round_trips_through_aggregation():
    solver = _solver()
    entries = build_forms(solver)
    cfgs = configurations(solver)
    transport = cfgs["TransportPropertiesConfig"].model_construct()

    fs = configs_to_form_state(entries, [transport])
    # Fill the rest of the map with empties so aggregation reads a full state.
    state = {e.key: fs.get(e.key, {}) for e in entries}
    spec = state_to_case_spec(entries, state, selected_models=set())
    assert "transport_properties_config" in spec


def test_models_filled_by_returns_optional_owner():
    solver = _solver()
    entries = build_forms(solver)
    cfgs = configurations(solver)
    boussinesq = cfgs["BoussinesqConfig"].model_construct()
    transport = cfgs["TransportPropertiesConfig"].model_construct()

    filled = models_filled_by(entries, [boussinesq, transport])
    assert filled == {
        "boussinesq"
    }  # transport is owned by required Newtonian → not returned
