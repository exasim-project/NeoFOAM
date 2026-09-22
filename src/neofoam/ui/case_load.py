# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Reopen an existing case: read it off disk and push it into the wizard forms.

Two halves of the one path every "open this case" route shares — the AI
assistant's ``load_case`` tool, the toolbar's "Load case" button and the sweep
panel's base-case restore — so all read the same configs and select the same models.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from neofoam.agent.case_fill import case_spec_to_configs, load_case_from_disk
from neofoam.framework.solver.configurations import configurations
from neofoam.io import OpenFOAMStrategy
from neofoam.io.dictread import Value, read_keys, read_section, read_toplevel
from neofoam.ui.case_spec import configs_to_form_state, models_filled_by
from neofoam.ui.forms import FormEntry
from neofoam.ui.steps import (
    ModelFamily,
    loaded_turbulence_model,
    select_model_state,
    selection_key,
)

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


#: Parses each dictionary named on its command line, the way every config load does.
_PARSE_PROBE = "import sys, pybFoam\nfor path in sys.argv[1:]:\n    pybFoam.dictionary.read(path)"


def _dictionary_files(case_dir: Path, solver: Any) -> list[str]:
    """The OpenFOAM dictionaries of ``case_dir`` that loading ``solver``'s configs parses."""
    metadata = (getattr(cls, "io_config", None) for cls in configurations(solver))
    names = {io.file for io in metadata if io and isinstance(io.reader, OpenFOAMStrategy)}
    return sorted(str(case_dir / name) for name in names if (case_dir / name).is_file())


def _require_parsable(case_dir: Path, solver: Any) -> None:
    """Raise ``ValueError`` with OpenFOAM's complaint when a dictionary does not parse."""
    # A parse error (stray brace, missing #include) is a FatalIOError: OpenFOAM exits
    # the process instead of raising, which would take the wizard server down with it.
    # So the files are parsed in a child first (~0.1 s); the parent reads what survived.
    probe = subprocess.run(
        [sys.executable, "-c", _PARSE_PROBE, *_dictionary_files(case_dir, solver)],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode:
        banner = (probe.stdout + probe.stderr).split("FOAM FATAL", 1)[-1]
        message = banner.split("\n", 1)[-1].split("    From ", 1)[0]
        reason, _, where = " ".join(message.split()).partition(" file: ")
        raise ValueError(f"{reason} ({where.rstrip('.')}).")


def read_case_configs(
    case_dir: Path,
    solver: Any,
    warnings: list[dict[str, str]] | None = None,
) -> list[Any]:
    """The configs an existing case carries on disk — deterministic, no LLM.

    Use this instead of asking the model to transcribe a case. Files that are
    absent are simply missing from the result; pass ``warnings`` to also collect
    the ones that are present but do not validate. A dictionary OpenFOAM cannot parse
    raises ``ValueError`` (found in a child process, as such an error ends the process
    it occurs in). Pair it with
    :func:`apply_configs_to_forms` to put the result in front of the user::

        configs = read_case_configs(Path(case_dir), solver)
    """
    _require_parsable(case_dir, solver)
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


def case_advection_model(case_dir: Path) -> str:
    """The alpha-advection model the VoF solver would run the case with.

    The same evidence the VoF solver's ``select_from_case`` runs on: the
    ``advectionScheme`` key, else isoAdvector-only controls in the alpha solver block.
    Without either — or without the file — it is ``MULES``, the solver's fallback::

        advection = case_advection_model(Path(case_dir))
    """
    fv_solution = case_dir / "system" / "fvSolution"
    named = read_toplevel(fv_solution, "advectionScheme")
    if isinstance(named, Value):
        return named.text
    blocks = read_section(fv_solution, "solvers").items()
    controls = {key for name, block in blocks if name.startswith("alpha.") for key in block}
    return "isoAdvector" if controls & _ISO_ADVECTOR_CONTROLS else "MULES"


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


def _clear_absent(
    state: Any,
    entries: list[FormEntry],
    families: list[ModelFamily],
    filled: dict[str, dict[str, Any]],
) -> None:
    """Reset the forms and switch off the optional models a loaded case does not fill.

    The replacement half of a load: without it the previous case's data stays live in
    every form the new one leaves out, and its optional models stay selected — so the
    next Save writes, say, the Boussinesq configs of the case before into the case just
    loaded. A form goes back to the defaults a fresh wizard seeds. A family member is
    never deselected: a pick-one family must always have exactly one member on, and
    which one is :func:`models_to_select`'s decision.
    """
    family_members = {choice.name for family in families for choice in family.members}
    filled_owners = {e.owner_model for e in entries if e.key in filled}
    for entry in entries:
        if entry.key in filled:
            continue
        state[entry.state_key] = dict(entry.defaults)
        owner = entry.owner_model
        if owner and owner not in filled_owners and owner not in family_members:
            state[selection_key(owner)] = False


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
    :func:`case_advection_model`) — and because that marks ``configs`` as one case's
    whole contents, the load then *replaces* instead of merging (:func:`_clear_absent`).
    Without a ``case_dir`` the configs are one incremental fill (the AI's own output)
    and the untouched forms stay as they are. Must run on
    trame's event loop like every other state write::

        selected = apply_configs_to_forms(state, entries, families, configs, case_dir)
    """
    by_key = {e.key: e for e in entries}
    filled = configs_to_form_state(entries, configs)
    for key, data in filled.items():
        state[by_key[key].state_key] = data
    if case_dir is not None:
        _clear_absent(state, entries, families, filled)
    algorithm = case_algorithm(case_dir, entries) if case_dir else None
    selected = models_to_select(models_filled_by(entries, configs), families, algorithm)
    for model in selected:
        # A member of a pick-one family (a loaded SIMPLE case) deselects its
        # siblings rather than joining them.
        state.update(select_model_state(families, model))
    # Neither turbulenceProperties nor the advection evidence has an owning model; only
    # a family member has a choice to move.
    advection = case_advection_model(case_dir) if case_dir else None
    applied = set(selected)
    for named in (loaded_turbulence_model(entries, state), advection):
        if any(c.name == named for family in families for c in family.members):
            state.update(select_model_state(families, str(named)))
            # Reported too: the caller's summary names every model the load moved.
            applied.add(str(named))
    return applied
