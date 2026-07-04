# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for the wizard step model + model choices."""

from __future__ import annotations

from neofoam.mcp.registry import resolve_solver
from neofoam.ui.forms import build_forms
from neofoam.ui.steps import build_model_choices, build_steps


def _solver():
    return resolve_solver("incompressibleFluid")


def test_steps_fixed_order_and_membership():
    solver = _solver()
    entries = build_forms(solver)
    steps = build_steps(solver, entries)

    assert [s.id for s in steps] == [
        "setup",
        "geometry",
        "models",
        "schemes",
        "bcs",
        "initial",
        "review",
    ]
    by_id = {s.id: s for s in steps}

    # setup / geometry / review are form-less bespoke panels.
    assert by_id["setup"].entry_keys == []
    assert by_id["geometry"].entry_keys == []
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


def test_model_choices_required_flags():
    choices = build_model_choices(_solver())
    by_name = {c.name: c for c in choices}
    assert by_name["Pimple"].required is True
    assert by_name["Newtonian"].required is True
    assert by_name["boussinesq"].required is False
    assert by_name["courant"].required is False
