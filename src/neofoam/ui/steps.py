# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Wizard step model + model-selection choices (headless, pure)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neofoam.mcp import tools
from neofoam.ui.forms import FormEntry

__all__ = ["Step", "ModelChoice", "build_steps", "build_model_choices"]

# Fixed order. `models` opens with the model-selection panel (rendered by app.py)
# above its config forms; `geometry` (STL → mesh dicts) and `review` hold no form
# entries — they are bespoke panels rendered by app.py.
_STEP_DEFS: list[tuple[str, str]] = [
    ("models", "Models"),
    ("geometry", "Geometry & mesh"),
    ("bcs", "Boundary conditions"),
    ("initial", "Initial values"),
    ("schemes", "Numerics"),
    ("review", "Review & run"),
]


@dataclass(frozen=True)
class Step:
    """One left-sidebar step and the form-entry keys shown under it."""

    id: str
    label: str
    entry_keys: list[str]


@dataclass(frozen=True)
class ModelChoice:
    """A selectable model: required ones are locked on."""

    name: str
    label: str
    required: bool


def build_steps(solver: Any, entries: list[FormEntry]) -> list[Step]:
    """The six steps in fixed order, each carrying its form-entry keys.

    Membership follows ``FormEntry.step`` (``models``/``schemes``/``bcs``/``initial``);
    ``geometry`` and ``review`` are form-less bespoke panels.
    """
    by_step: dict[str, list[str]] = {sid: [] for sid, _ in _STEP_DEFS}
    for entry in entries:
        if entry.step in by_step:
            by_step[entry.step].append(entry.key)
    return [Step(sid, label, by_step[sid]) for sid, label in _STEP_DEFS]


def build_model_choices(solver: Any) -> list[ModelChoice]:
    """Required (locked) + optional (add/remove) models from the catalog."""
    return [
        ModelChoice(name=e.name, label=e.label, required=e.required)
        for e in tools.model_catalog(solver)
    ]
