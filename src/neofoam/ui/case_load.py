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
import warnings as python_warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pybFoam as pyf
from pydantic import RootModel, ValidationError

from neofoam.agent.case_fill import (
    _file_keys,
    _holds_any,
    _own_keys,
    case_spec_to_configs,
    load_case_from_disk,
)
from neofoam.agent.case_forms import is_scheme_config
from neofoam.foam import LinearSolverConversionWarning
from neofoam.framework.solver.configurations import configurations
from neofoam.io import OpenFOAMStrategy
from neofoam.io.dictread import Value, foam_case, read_keys, read_section, read_toplevel
from neofoam.io.strategies.openfoam_strategy import _read_fields, _to_python
from neofoam.ui.case_spec import configs_to_form_state, models_filled_by
from neofoam.ui.form_schema import NF_FIELD
from neofoam.ui.forms import FormEntry
from neofoam.ui.steps import (
    ModelFamily,
    loaded_turbulence_model,
    select_model_state,
    selection_key,
)

__all__ = [
    "read_case_configs",
    "conversion_notes",
    "apply_configs_to_forms",
    "remember_rejected_slices",
    "rejected_slices",
    "without_rejected_files",
    "undeclared_entries_note",
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
    it occurs in). A warning whose missing keys are valid regex entries of the file
    (``"(U|k)"``) gets ``"cause": "regex"``. Pair it with
    :func:`apply_configs_to_forms` to put the result in front of the user::

        configs = read_case_configs(Path(case_dir), solver)
    """
    # Both reads under one $FOAM_CASE: the probe has to see exactly what the real read
    # will, or a case whose dicts use OpenFOAM's <system> tag passes the probe and then
    # aborts the server process on the read it was meant to protect.
    with foam_case(case_dir):
        _require_parsable(case_dir, solver)
        configs = case_spec_to_configs(
            load_case_from_disk(case_dir, solver=solver, warnings=warnings)
        )
        # Here, not at save: the file has just passed the parse probe.
        if warnings is not None:
            _note_regex_slices(case_dir, solver, configs, warnings)
        return configs


def conversion_notes(case_dir: Path, configs: list[Any]) -> list[str]:
    """One line per loaded solver block whose conversion dropped keys of the file.

    Use it after :func:`read_case_configs` to report what a legacy block lost. Each block
    is converted again here because the warning itself does not name its field. Warnings
    filters are process-global, so this must run on trame's event loop::

        notes = conversion_notes(Path(case_dir), configs)
    """
    notes: dict[str, None] = {}
    with foam_case(case_dir):
        for config in configs:
            for field, message in _block_conversions(case_dir, config):
                notes[f"{type(config).io_config.file} solvers.{field}: {message}"] = None
    return list(notes)


def _block_conversions(case_dir: Path, config: Any) -> list[tuple[str, str]]:
    """The ``(field, warning)`` pairs of converting ``config``'s typed solver blocks."""
    solvers = getattr(config, "solvers", None)
    typed = {
        info.alias or name: type(getattr(solvers, name))
        for name, info in (type(solvers).model_fields.items() if solvers else ())
        if isinstance(getattr(solvers, name), RootModel)
    }
    if not typed:
        return []
    raw = _to_python(pyf.dictionary.read(str(case_dir / config.io_config.file)))["solvers"]
    found: list[tuple[str, str]] = []
    for field in typed.keys() & raw.keys():
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always", LinearSolverConversionWarning)
            typed[field].model_validate(raw[field])
        found += [(field, str(w.message)) for w in caught]
    return sorted(found)


def remember_rejected_slices(state: Any, case_dir: Path, warnings: list[dict[str, str]]) -> None:
    """Record the slices a load found on disk but could not validate.

    Their forms hold defaults rather than the file's values, so every writer asks
    :func:`rejected_slices` before touching the case. Call it after each successful
    read; a new read replaces the record::

        remember_rejected_slices(state, Path(case_dir), warnings)
    """
    where = str(case_dir.resolve())
    state.rejected_slices = [{**warning, "case_dir": where} for warning in warnings]


def rejected_slices(
    state: Any, entries: list[FormEntry], target: Path
) -> dict[str, list[dict[str, str]]]:
    """The files in ``target`` a writer must leave alone, each with its rejected slices.

    Only a slice a save would write counts: one with a form, whose model is selected,
    in the case it was read from, and whose file is still there. A slice whose missing
    keys are valid regex entries counts whichever model is selected, because any write
    of that file would clear them, and is marked ``"cause": "regex"``::

        refused = rejected_slices(state, entries, target)
    """
    by_name = {entry.cls_name: entry for entry in entries}
    where = str(target.resolve())
    refused: dict[str, list[dict[str, str]]] = {}
    for rejected in state.rejected_slices or []:
        entry, file = by_name.get(rejected["config"]), rejected["file"]
        if rejected["case_dir"] != where or entry is None or not (target / file).exists():
            continue
        regex = rejected.get("cause") == "regex"
        if rejected.get("cause") == "partial" and entry.owner_model is None:
            continue
        if regex or entry.owner_model is None or state[selection_key(entry.owner_model)]:
            refused.setdefault(file, []).append(rejected)
    return refused


def _note_regex_slices(
    case_dir: Path, solver: Any, configs: list[Any], warnings: list[dict[str, str]]
) -> None:
    """Mark reported slices that regex keys complete and report every other regex file.

    The load takes a slice without a key of its own for absent, yet a form default
    written over its file would clear the regex block. A slice that did load is no
    safer: the writer drops the key's quotes, which leaves a file OpenFOAM cannot parse.
    """
    classes = {cls.__name__: cls for cls in configurations(solver)}
    for warning in warnings:
        cls = classes[warning["config"]]
        if _regex_cause(cls, case_dir / warning["file"], warning["reason"]) == "regex":
            warning["cause"] = "regex"
    settled = {type(config).__name__ for config in configs} | {w["config"] for w in warnings}
    for name, cls in classes.items():
        io = getattr(cls, "io_config", None)
        if name in settled or io is None or not (case_dir / io.file).exists():
            continue
        reason = _missing_keys_reason(cls, case_dir)
        cause = _regex_cause(cls, case_dir / io.file, reason) if reason else None
        noted = {"file": io.file, "config": name, "reason": reason or ""}
        if reason and cause:
            warnings.append({**noted, "cause": "regex"} if cause == "regex" else noted)
        elif reason and _is_unjudged_partial(cls, list(classes.values()), case_dir / io.file):
            warnings.append({**noted, "cause": "partial"})
    _note_loaded_regex_files(case_dir, configs, warnings)


def _is_unjudged_partial(cls: Any, classes: list[Any], path: Path) -> bool:
    """True for a slice holding some keys that rivals with the same keys hide from the load.

    ``kEpsilon`` and ``realizableKE`` require the same entries, so neither has a key of its
    own. Only the selected one matters, which :func:`rejected_slices` decides at save.
    """
    return not _own_keys(cls, classes) and _holds_any(path, _file_keys(cls))


def _note_loaded_regex_files(
    case_dir: Path, configs: list[Any], warnings: list[dict[str, str]]
) -> None:
    """Report each loaded config whose file spells a key as a quoted regex."""
    reported = {w["file"] for w in warnings}
    for config in configs:
        io = getattr(type(config), "io_config", None)
        if io is None or io.file in reported or not (case_dir / io.file).is_file():
            continue
        text = (case_dir / io.file).read_text(errors="ignore")
        entries = (
            _regex_entries(pyf.dictionary.read(str(case_dir / io.file)), text)
            if '"' in text
            else []
        )
        if entries:
            reason = "regex keys " + ", ".join(entries)
            warnings.append(
                {
                    "file": io.file,
                    "config": type(config).__name__,
                    "reason": reason,
                    "cause": "regex",
                }
            )


def _regex_entries(node: Any, text: str, prefix: str = "") -> list[str]:
    """The dotted keys below ``node`` that the file's ``text`` spells in quotes."""
    found: list[str] = []
    for entry in node.toc():
        key = str(entry)
        if f'"{key}"' in text:
            found.append(prefix + key)
        elif node.isDict(key):
            found += _regex_entries(node.subDict(key), text, f"{prefix}{key}.")
    return found


def _missing_keys_reason(cls: Any, case_dir: Path) -> str | None:
    """``missing <key>, <key>`` when only absent keys keep ``cls`` from loading."""
    try:
        cls.load(case_dir=case_dir)
    except KeyError:
        return None
    except ValidationError as exc:
        errors = exc.errors()
        if all(error["type"] == "missing" for error in errors):
            return "missing " + ", ".join(".".join(map(str, e["loc"])) for e in errors)
    return None


def _regex_cause(cls: Any, path: Path, reason: str) -> str | None:
    """``regex`` when the file's regex keys validly supply every missing key.

    ``invalid`` when they supply them but the slice still fails, else ``None``.
    """
    if not reason.startswith("missing ") or cls.io_config.subdict:
        return None
    root = pyf.dictionary.read(str(path))
    data = _to_python(root)
    for loc in reason.removeprefix("missing ").split(", "):
        if not _copy_matched_entry(root, data, loc.split(".")):
            return None
    try:
        cls.model_validate(_read_fields(cls, data))
    except ValidationError:
        return "invalid"
    return "regex"


def _copy_matched_entry(root: Any, data: dict[str, Any], keys: list[str]) -> bool:
    """Copy the entry OpenFOAM's pattern lookup finds for ``keys`` into ``data``."""
    *parents, leaf = keys
    node, section = root, data
    for part in parents:
        if not node.isDict(part):
            return False
        node, section = node.subDict(part), section[part]
    if not node.found(leaf):
        return False
    section[leaf] = (
        _to_python(node.subDict(leaf)) if node.isDict(leaf) else str(node.get[str](leaf))
    )
    return True


def without_rejected_files(
    spec: dict[str, Any],
    entries: list[FormEntry],
    refused: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    """``spec`` minus every config that writes to a ``refused`` file, co-owners included."""
    file_of = {e.config_name: e.cls.io_config.file for e in entries if _has_file(e)}
    return {name: data for name, data in spec.items() if file_of.get(name) not in refused}


def _field_sections(entry: FormEntry) -> dict[str, set[str]]:
    """The sections of a per-field form holding field entries, with their declared keys."""
    sections = entry.form_schema.get("properties", {}).items()
    return {
        name: set(section["properties"])
        for name, section in sections
        if any(NF_FIELD in item for item in section.get("properties", {}).values())
    }


def undeclared_entries_note(state: Any, entries: list[FormEntry], configs: list[Any]) -> list[str]:
    """One line per file naming the loaded entries no selected model declares.

    The per-field forms draw declared entries only; these stay in the data and are
    written back unchanged. ``default`` is left out: it backs the declared entries.
    Call it after :func:`apply_configs_to_forms`, which settles the selection::

        notes = undeclared_entries_note(state, entries, configs)
    """
    active = {
        e.cls_name: e
        for e in entries
        if e.field_schema and (e.owner_model is None or state[selection_key(e.owner_model)])
    }
    declared: dict[str, set[tuple[str, str]]] = {}
    for entry in active.values():
        pairs = {(s, key) for s, keys in _field_sections(entry).items() for key in keys}
        declared.setdefault(entry.cls.io_config.file, set()).update(pairs)
    hidden: dict[str, dict[str, None]] = {}
    for config in configs:
        form = active.get(type(config).__name__)
        if form is None:
            continue
        file = form.cls.io_config.file
        dump = config.model_dump(by_alias=True, exclude_none=True)
        for section in _field_sections(form):
            for key in dump.get(section, {}):
                if key != "default" and (section, key) not in declared[file]:
                    hidden.setdefault(file, {})[f"{section}.{key}"] = None
    return [
        f"{file}: not used by the selected models, hidden and kept as written: "
        + ", ".join(f"`{name}`" for name in names)
        + "."
        for file, names in hidden.items()
    ]


def _has_file(entry: FormEntry) -> bool:
    """True when ``entry``'s config is bound to a file."""
    return getattr(entry.cls, "io_config", None) is not None


def case_algorithm(case_dir: Path, entries: list[FormEntry]) -> str | None:
    """The pressure-velocity model the case's ``system/fvSolution`` control block names.

    The same evidence the solver's ``detect_and_create`` runs on, so the wizard opens a
    case on the algorithm it would run with. ``None`` when the file has no control block
    or no selectable model declares it (a solver with a single algorithm)::

        algorithm = case_algorithm(Path(case_dir), entries)
    """
    with foam_case(case_dir):
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
    with foam_case(case_dir):
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
    evidenced: set[str],
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
    for entry in entries:
        if entry.key in filled:
            continue
        state[entry.state_key] = dict(entry.defaults)
        owner = entry.owner_model
        if owner and owner not in evidenced and owner not in family_members:
            state[selection_key(owner)] = False


def _models_in_case(entries: list[FormEntry], configs: list[Any], case_dir: Path) -> set[str]:
    """The models the case's files show, not counting a slice only ``default`` completes.

    Every operator of an all-``default`` ``fvSchemes`` validates, so that slice is valid
    for any model. It keeps its values for the form, but selects nothing.
    """
    classes = list(dict.fromkeys(e.cls for e in entries if _has_file(e)))
    shown = []
    for config in configs:
        cls = type(config)
        own = _own_keys(cls, classes) if is_scheme_config(cls) else set()
        if not own or _holds_any(case_dir / cls.io_config.file, own):
            shown.append(config)
    return models_filled_by(entries, shown)


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
    filled_models = (
        models_filled_by(entries, configs)
        if case_dir is None
        else _models_in_case(entries, configs, case_dir)
    )
    if case_dir is not None:
        _clear_absent(state, entries, families, filled, filled_models)
    # JSONForms keeps a combinator's selected arm across a data replace, so a loaded
    # block would show the previous arm over the new data until the form remounts.
    state.form_epoch = (state.form_epoch or 0) + 1
    algorithm = case_algorithm(case_dir, entries) if case_dir else None
    selected = models_to_select(filled_models, families, algorithm)
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
