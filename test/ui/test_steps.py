# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for the wizard step model + model choices."""

from __future__ import annotations

import pytest

from neofoam.mcp.registry import resolve_solver
from neofoam.ui.forms import build_forms
from neofoam.ui.steps import (
    build_model_choices,
    build_model_families,
    build_steps,
    choice_key,
    select_model_state,
    selection_key,
)


def test_steps_fixed_order_and_membership(solver):
    entries = build_forms(solver)
    steps = build_steps(solver, entries)

    assert [s.id for s in steps] == [
        "models",
        "geometry",
        "bcs",
        "initial",
        "schemes",
        "sweep",
        "review",
    ]
    by_id = {s.id: s for s in steps}

    # geometry / sweep / review are form-less bespoke panels.
    assert by_id["geometry"].entry_keys == []
    assert by_id["sweep"].entry_keys == []
    assert by_id["review"].entry_keys == []

    # Each form step holds exactly the entries whose FormEntry.step matches.
    for sid in ("models", "schemes", "bcs", "initial"):
        expected = {e.key for e in entries if e.step == sid}
        assert set(by_id[sid].entry_keys) == expected

    # bcs holds field_bc halves; initial holds field_in halves.
    assert all(k.startswith("field_bc:") for k in by_id["bcs"].entry_keys)
    assert all(k.startswith("field_in:") for k in by_id["initial"].entry_keys)
    # schemes are fv* dict configs.
    assert all(k.startswith("dict:") for k in by_id["schemes"].entry_keys)


def test_model_choices_required_flags(solver):
    choices = build_model_choices(solver)
    by_name = {c.name: c for c in choices}
    assert by_name["Pimple"].required is True
    assert by_name["Newtonian"].required is True
    assert by_name["boussinesq"].required is False
    assert by_name["courant"].required is False


def test_model_families_are_the_pick_one_choices(solver):
    families = {f.name: f for f in build_model_families(solver)}
    assert set(families) == {"PressureVelocityAlgorithm", "momentumTransportModel"}
    algorithm = families["PressureVelocityAlgorithm"]
    assert algorithm.label == "Pressure Velocity Algorithm"
    assert [c.name for c in algorithm.members] == ["Pimple", "Simple"]
    assert all(c.required for c in algorithm.members)


def test_select_model_state_turns_the_siblings_off(solver):
    families = build_model_families(solver)

    updates = select_model_state(families, "Simple")

    assert updates == {
        "sel_Simple": True,
        "sel_Pimple": False,
        "choice_PressureVelocityAlgorithm": "Simple",
    }


def test_select_model_state_leaves_a_toggle_alone(solver):
    families = build_model_families(solver)
    # An optional model belongs to no pick-one family — nothing else moves.
    assert select_model_state(families, "boussinesq") == {"sel_boussinesq": True}


@pytest.mark.parametrize("solver_name", ["incompressibleFluid", "incompressibleFluidNeoN"])
def test_time_step_models_carry_a_display_label(solver_name):
    choices = {c.name: c for c in build_model_choices(resolve_solver(solver_name))}

    assert choices["courant"].label == "Adaptive time step (Courant)"
    assert choices["maxDeltaT"].label == "Time step limit (maxDeltaT)"


@pytest.mark.parametrize(
    ("make_key", "name", "expected"),
    [
        # Identifier names keep today's state names (a de-facto API for tests/examples).
        (selection_key, "kEpsilon", "sel_kEpsilon"),
        (choice_key, "momentumTransportModel", "choice_momentumTransportModel"),
        # A `-` or `.` would otherwise break the Vue expression the key is used in.
        (selection_key, "k-omega.SST", "sel_k_omega_SST"),
        (choice_key, "my-family", "choice_my_family"),
    ],
)
def test_model_state_keys_are_js_identifiers(make_key, name, expected):
    assert make_key(name) == expected
