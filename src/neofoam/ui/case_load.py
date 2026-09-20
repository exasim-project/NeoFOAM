# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Reopen an existing case: read it off disk and push it into the wizard forms.

Two halves of the one path every "open this case" route shares — the AI
assistant's ``load_case`` tool and the sweep panel's base-case restore — so both
read the same configs and select the same models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neofoam.agent.case_fill import case_spec_to_configs, load_case_from_disk
from neofoam.ui.case_spec import configs_to_form_state, models_filled_by
from neofoam.ui.forms import FormEntry
from neofoam.ui.steps import ModelFamily, loaded_turbulence_model, select_model_state

__all__ = ["read_case_configs", "apply_configs_to_forms"]


def read_case_configs(
    case_dir: Path,
    solver: Any,
    warnings: list[dict[str, str]] | None = None,
) -> list[Any]:
    """The configs an existing case carries on disk — deterministic, no LLM.

    Use this instead of asking the model to transcribe a case. Files that are
    absent are simply missing from the result; pass ``warnings`` to also collect
    the ones that are present but do not validate. Pair it with
    :func:`apply_configs_to_forms` to put the result in front of the user::

        configs = read_case_configs(Path(case_dir), solver)
    """
    return case_spec_to_configs(load_case_from_disk(case_dir, solver=solver, warnings=warnings))


def apply_configs_to_forms(
    state: Any,
    entries: list[FormEntry],
    families: list[ModelFamily],
    configs: list[Any],
) -> set[str]:
    """Push ``configs`` into the live form state; returns the models they filled.

    Overwrites whatever the forms currently hold for those configs, so a caller
    that can lose user edits asks first. Must run on trame's event loop like
    every other state write::

        filled = apply_configs_to_forms(state, entries, families, configs)
    """
    by_key = {e.key: e for e in entries}
    for key, data in configs_to_form_state(entries, configs).items():
        state[by_key[key].state_key] = data
    filled_models = models_filled_by(entries, configs)
    for model in filled_models:
        # A filled member of a pick-one family (a loaded SIMPLE case) deselects
        # its siblings rather than joining them.
        state.update(select_model_state(families, model))
    # turbulenceProperties has no owning model; only a family member has a choice to move.
    turbulence = loaded_turbulence_model(entries, state)
    if any(c.name == turbulence for family in families for c in family.members):
        state.update(select_model_state(families, str(turbulence)))
    return filled_models
