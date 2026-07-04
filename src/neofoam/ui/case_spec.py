# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Aggregate live form state into a ``save_case`` spec dict (headless, pure).

The correctness core of the wizard: read each form's live data object, merge the two
halves of a field config, drop empty forms and forms owned by an unselected optional
model, and key everything by snake-case class name — exactly the dict
:func:`neofoam.mcp.tools.save_case` validates and writes.
"""

from __future__ import annotations

from typing import Any

from neofoam.agent.case_forms import merge_field_config, split_field_dump
from neofoam.ui.forms import FormEntry

__all__ = [
    "owned_entry_keys_by_model",
    "state_to_case_spec",
    "configs_to_form_state",
    "models_filled_by",
]


def _defaults_complete(cls: Any) -> bool:
    """True when ``cls()`` validates — every field has a default.

    Such a config (e.g. ``fvSchemes``/``fvSolution``) is written even when its form
    was never touched: the case needs the file, and the class supplies full defaults.
    """
    if cls is None:
        return False
    try:
        cls()
    except Exception:  # noqa: BLE001 - any validation error ⇒ user input required
        return False
    return True


def owned_entry_keys_by_model(entries: list[FormEntry]) -> dict[str, list[str]]:
    """Optional-model name → the form-entry keys it owns (for hide/skip)."""
    owned: dict[str, list[str]] = {}
    for entry in entries:
        if entry.owner_model:
            owned.setdefault(entry.owner_model, []).append(entry.key)
    return owned


def state_to_case_spec(
    entries: list[FormEntry],
    form_state: dict[str, dict[str, Any]],
    selected_models: set[str],
) -> dict[str, dict[str, Any]]:
    """Aggregate ``form_state`` (entry.key → live data) into the save_case dict.

    - dict entry: taken whole; an *empty* form is still included when its config
      class default-constructs (the case needs ``system/fvSchemes`` etc. even if the
      user never opened that panel); skipped when empty **and** requiring input, or
      when owned by an unselected model.
    - field config: input + BC halves merged via
      :func:`neofoam.agent.case_forms.merge_field_config`; skipped if both halves are
      empty or the config is owned by an unselected model.
    - keyed by snake-case class name (a ``build_case_output_model`` field).
    """
    spec: dict[str, dict[str, Any]] = {}
    # config_name -> {"cls", "in", "bc"} for field halves that survive filtering.
    fields: dict[str, dict[str, Any]] = {}

    for entry in entries:
        if entry.owner_model and entry.owner_model not in selected_models:
            continue
        data = form_state.get(entry.key) or {}
        if entry.kind == "dict":
            if data or _defaults_complete(entry.cls):
                spec[entry.config_name] = data
            continue
        slot = fields.setdefault(
            entry.config_name, {"cls": entry.cls, "in": None, "bc": None}
        )
        if entry.kind == "field_in":
            slot["in"] = data or None
        else:
            slot["bc"] = data or None

    for config_name, slot in fields.items():
        if not slot["in"] and not slot["bc"]:
            continue
        merged = merge_field_config(slot["cls"], slot["in"], slot["bc"])
        spec[config_name] = merged.model_dump(by_alias=True, exclude_none=True)

    return spec


def configs_to_form_state(
    entries: list[FormEntry],
    configs: list[Any],
) -> dict[str, dict[str, Any]]:
    """Reverse mapping: filled config instances → per-entry live data (entry.key → data).

    A dict config maps to its ``dict:<Cls>`` entry; a field config is split via
    :func:`neofoam.agent.case_forms.split_field_dump` into its ``field_in``/``field_bc``
    halves. Configs with no matching entry are ignored. Used by the AI-fill push.
    """
    by_cls: dict[str, dict[str, FormEntry]] = {}
    for entry in entries:
        by_cls.setdefault(entry.cls_name, {})[entry.kind] = entry

    out: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        kinds = by_cls.get(type(cfg).__name__)
        if not kinds:
            continue
        dump = cfg.model_dump(by_alias=True, exclude_none=True)
        if "dict" in kinds:
            out[kinds["dict"].key] = dump
        else:
            in_half, bc_half = split_field_dump(dump)
            if "field_in" in kinds:
                out[kinds["field_in"].key] = in_half
            if "field_bc" in kinds:
                out[kinds["field_bc"].key] = bc_half
    return out


def models_filled_by(entries: list[FormEntry], configs: list[Any]) -> set[str]:
    """Optional-model names that own any of the filled configs (for auto-select)."""
    owner_by_cls = {e.cls_name: e.owner_model for e in entries}
    filled: set[str] = set()
    for cfg in configs:
        owner = owner_by_cls.get(type(cfg).__name__)
        if owner:
            filled.add(owner)
    return filled
