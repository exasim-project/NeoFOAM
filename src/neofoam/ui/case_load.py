# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Reopen an existing case: read it off disk and push it into the wizard forms.

Two halves of the one path every "open this case" route shares — the AI
assistant's ``load_case`` tool, the toolbar's "Load case" button and the sweep
panel's base-case restore — so all read the same configs and select the same models.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from neofoam.agent.case_fill import case_spec_to_configs, load_case_from_disk
from neofoam.io.dictread import Value, read_keys, read_section, read_toplevel
from neofoam.ui.case_spec import configs_to_form_state, models_filled_by
from neofoam.ui.forms import FormEntry
from neofoam.ui.steps import ModelFamily, loaded_turbulence_model, select_model_state

__all__ = [
    "read_case_configs",
    "apply_configs_to_forms",
    "case_algorithm",
    "case_advection_model",
    "models_to_select",
]

#: ``system/fvSolution`` control block → the block the owning member's form declares,
#: in the order ``detect_and_create`` looks for them (PISO is a single-outer-loop PIMPLE).
_CONTROL_BLOCKS = {"PIMPLE": "PIMPLE", "SIMPLE": "SIMPLE", "PISO": "PIMPLE"}

#: Keys only an isoAdvector ``solvers`` "alpha.*" block holds — the evidence the VoF
#: solver's ``select_from_case`` falls back on when ``advectionScheme`` is absent.
_ISO_ADVECTOR_CONTROLS = {"reconstructionScheme", "isoFaceTol", "surfCellTol", "nAlphaBounds"}


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


def case_algorithm(case_dir: Path, entries: list[FormEntry]) -> str | None:
    """The pressure-velocity model the case's ``system/fvSolution`` control block names.

    The same evidence the solver's ``detect_and_create`` runs on, so the wizard opens a
    case on the algorithm it would run with. ``None`` when the file has no control block
    or no selectable model declares it (a solver with a single algorithm)::

        algorithm = case_algorithm(Path(case_dir), entries)
    """
    keys = read_keys(case_dir / "system" / "fvSolution")
    if not isinstance(keys, frozenset):
        return None
    declared = next((block for name, block in _CONTROL_BLOCKS.items() if name in keys), None)
    owners = (e.owner_model for e in entries if declared in e.schema.get("properties", {}))
    return next((owner for owner in owners if owner), None)


def case_advection_model(case_dir: Path) -> str | None:
    """The alpha-advection model the case's ``system/fvSolution`` is evidence of.

    The same evidence the VoF solver's ``select_from_case`` runs on: the
    ``advectionScheme`` key, else isoAdvector-only controls in the alpha solver block.
    ``None`` without either, so the wizard then keeps its current choice::

        advection = case_advection_model(Path(case_dir))
    """
    fv_solution = case_dir / "system" / "fvSolution"
    named = read_toplevel(fv_solution, "advectionScheme")
    if isinstance(named, Value):
        return named.text
    blocks = read_section(fv_solution, "solvers").items()
    controls = {key for name, block in blocks if name.startswith("alpha.") for key in block}
    return "isoAdvector" if controls & _ISO_ADVECTOR_CONTROLS else None


def models_to_select(
    filled: Iterable[str], families: list[ModelFamily], algorithm: str | None
) -> list[str]:
    """The filled models a load switches on, in a fixed order.

    A pick-one family contributes one member at most: ``algorithm`` when it is one of
    its members, else its only filled member. Several filled members are no evidence —
    a PIMPLE case's fvSchemes validate as the Simple slice too — so the family then
    keeps the wizard's current choice::

        models_to_select({"Pimple", "Simple", "boussinesq"}, families, "Pimple")
    """
    member_names = [{choice.name for choice in family.members} for family in families]
    chosen = {name for name in filled if not any(name in names for names in member_names)}
    for names in member_names:
        rivals = names.intersection(filled)
        if algorithm in names:
            chosen.add(str(algorithm))
        elif len(rivals) == 1:
            chosen |= rivals
    return sorted(chosen)


def apply_configs_to_forms(
    state: Any,
    entries: list[FormEntry],
    families: list[ModelFamily],
    configs: list[Any],
    case_dir: Path | None = None,
) -> set[str]:
    """Push ``configs`` into the live form state; returns the models it selected.

    Overwrites whatever the forms currently hold for those configs, so a caller
    that can lose user edits asks first. Pass the ``case_dir`` the configs were read
    from so a pick-one family follows the case (:func:`case_algorithm`,
    :func:`case_advection_model`). Must run on
    trame's event loop like every other state write::

        selected = apply_configs_to_forms(state, entries, families, configs, case_dir)
    """
    by_key = {e.key: e for e in entries}
    for key, data in configs_to_form_state(entries, configs).items():
        state[by_key[key].state_key] = data
    algorithm = case_algorithm(case_dir, entries) if case_dir else None
    selected = models_to_select(models_filled_by(entries, configs), families, algorithm)
    for model in selected:
        # A member of a pick-one family (a loaded SIMPLE case) deselects its
        # siblings rather than joining them.
        state.update(select_model_state(families, model))
    # Neither turbulenceProperties nor the advection evidence has an owning model; only
    # a family member has a choice to move.
    advection = case_advection_model(case_dir) if case_dir else None
    for named in (loaded_turbulence_model(entries, state), advection):
        if any(c.name == named for family in families for c in family.members):
            state.update(select_model_state(families, str(named)))
    return set(selected)
