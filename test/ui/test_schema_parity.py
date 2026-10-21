# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Every value the wizard pushes into a form must validate against that form's schema.

JSONForms receives ``entry.schema`` and the live form data together. When they disagree
it does not fail loudly — it renders the *wrong value*: a plain string ``"Gauss upwind"``
offered against an object ``oneOf`` matches no arm, so the combinator renderer falls back
to arm 0 and the field displays ``none``. The case on disk stays right while the screen
shows something else, which is worse than a crash.

The cause is configs that serialise to OpenFOAM syntax (``"Gauss upwind"``,
``"(water air)"``, ``"(0 -9.81 0)"``) while ``model_json_schema()`` publishes the
structured shape. These tests pin both ends of the wizard's data flow — the seeded
defaults and a real case read off disk — for every registered solver.

``required`` errors are excluded deliberately: a partly-filled form is the normal state
of the wizard (a pristine ``controlDict`` has no ``endTime``), and completeness is
``validate_case``'s job, not the renderer's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pybFoam")
jsonschema = pytest.importorskip("jsonschema")  # transitive via the `mcp` extra

from neofoam.agent.case_fill import case_spec_to_configs, load_case_from_disk  # noqa: E402
from neofoam.mcp.registry import list_solver_names, resolve_solver  # noqa: E402
from neofoam.ui.case_spec import configs_to_form_state  # noqa: E402
from neofoam.ui.forms import FormEntry, build_forms  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: A real on-disk case per solver, so the load path is checked against real values and
#: not only against the pre-seeded defaults.
_CASES = {
    "incompressibleFluid": _REPO_ROOT / "test/solver/incompressibleFluid/val_pitzDaily",
    "incompressibleFluidNeoN": _REPO_ROOT
    / "test/solver/incompressibleFluidNeoN/cases/pitzDailySteady",
    "incompressibleVoF": _REPO_ROOT / "test/solver/incompressibleVoF/cases/damBreak",
}

#: Mismatches that are known and not yet fixed, as
#: ``(stage, solver, form title, top-level property, failing keyword)``.
#:
#: This is a ledger, not a suppression list. Each test asserts the mismatches it finds
#: are *exactly* these, so an unlisted mismatch fails — and so does a listed one that
#: has been fixed, whose line must then be deleted. Keep it shrinking.
_OUTSTANDING = {
    # The boundary-condition schema pairs a catch-all "fallback BC arm" with the
    # specific per-type arms, so a plain `{"type": "noSlip"}` is valid under *both* and
    # violates `oneOf`'s exactly-one rule. Distinct from the cases above: the data and
    # the schema agree, the schema is merely ambiguous (`anyOf` would model it).
    ("loaded", "incompressibleFluidNeoN", "U — boundary conditions", "boundaryField", "oneOf"),
    ("loaded", "incompressibleFluidNeoN", "p — boundary conditions", "boundaryField", "oneOf"),
}


def _mismatches(
    stage: str, solver_name: str, entry: FormEntry, data: dict[str, Any]
) -> dict[tuple[str, ...], str]:
    """Schema violations in ``data`` that are not mere absence, keyed for the ledger."""
    found: dict[tuple[str, ...], str] = {}
    for err in jsonschema.Draft202012Validator(entry.schema).iter_errors(data):
        if err.validator == "required":
            continue
        prop = str(next(iter(err.absolute_path), "<root>"))
        key = (stage, solver_name, entry.title, prop, str(err.validator))
        path = "/".join(str(p) for p in err.absolute_path) or "<root>"
        found.setdefault(key, f"{path}: {err.message}")
    return found


def _assert_ledger(found: dict[tuple[str, ...], str], stage: str, solver_name: str) -> None:
    scope = {k for k in _OUTSTANDING if k[0] == stage and k[1] == solver_name}
    new = {k: v for k, v in found.items() if k not in scope}
    assert not new, f"new schema mismatches (form data does not match its schema): {new}"
    fixed = scope - set(found)
    assert not fixed, f"these mismatches are fixed — delete them from _OUTSTANDING: {sorted(fixed)}"


@pytest.mark.parametrize("solver_name", list_solver_names())
def test_seeded_defaults_validate_against_their_form_schema(solver_name: str) -> None:
    found: dict[tuple[str, ...], str] = {}
    for entry in build_forms(resolve_solver(solver_name)):
        found.update(_mismatches("defaults", solver_name, entry, entry.defaults))
    _assert_ledger(found, "defaults", solver_name)


@pytest.mark.parametrize("solver_name", sorted(_CASES))
def test_loaded_case_validates_against_its_form_schema(solver_name: str) -> None:
    solver = resolve_solver(solver_name)
    entries = build_forms(solver)
    by_key = {e.key: e for e in entries}

    configs = case_spec_to_configs(load_case_from_disk(_CASES[solver_name], solver=solver))
    form_state = configs_to_form_state(entries, configs)
    assert form_state, f"{solver_name}: fixture case loaded no configs — check the path"

    found: dict[tuple[str, ...], str] = {}
    for key, data in form_state.items():
        found.update(_mismatches("loaded", solver_name, by_key[key], data))
    _assert_ledger(found, "loaded", solver_name)
