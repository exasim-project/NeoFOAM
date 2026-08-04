# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Wizard step model + model-selection choices (headless, pure)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from neofoam.mcp import tools
from neofoam.ui.forms import FormEntry
from neofoam.ui.plugins import AT_START, StepPlugin

__all__ = ["Step", "ModelChoice", "build_steps", "build_model_choices"]

# Fixed order. `models` opens with the model-selection panel (rendered by app.py)
# above its config forms; `geometry` (STL → mesh dicts), `sweep` (the parameter
# canvas) and `review` hold no form entries — they are bespoke panels rendered
# by app.py.
_STEP_DEFS: list[tuple[str, str]] = [
    ("models", "Models"),
    ("geometry", "Geometry & mesh"),
    ("bcs", "Boundary conditions"),
    ("initial", "Initial values"),
    ("schemes", "Numerics"),
    ("sweep", "Parameters"),
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


def build_steps(
    solver: Any,
    entries: list[FormEntry],
    plugins: Sequence[StepPlugin] = (),
) -> list[Step]:
    """The built-in steps in fixed order, with any ``plugins`` woven in.

    Membership follows ``FormEntry.step`` (``models``/``schemes``/``bcs``/``initial``);
    ``geometry``/``sweep``/``review`` are form-less bespoke panels. Each plugin adds
    one extra step, slotted after its ``after`` anchor (an existing step id) or
    appended when the anchor is absent; ``plugins`` is assumed pre-ordered
    (see :func:`neofoam.ui.plugins.order_steps`). A plugin's form-entry keys follow
    ``FormEntry.step == plugin.id`` (usually empty — plugins are bespoke panels).
    """
    step_ids = {sid for sid, _ in _STEP_DEFS} | {p.id for p in plugins}
    by_step: dict[str, list[str]] = {sid: [] for sid in step_ids}
    for entry in entries:
        if entry.step in by_step:
            by_step[entry.step].append(entry.key)
    steps = [Step(sid, label, by_step[sid]) for sid, label in _STEP_DEFS]

    # `AT_START` plugins go first (before every built-in), in discovery order.
    at_start = [p for p in plugins if p.after == AT_START]
    for offset, plugin in enumerate(at_start):
        steps.insert(offset, Step(plugin.id, plugin.label, by_step.get(plugin.id, [])))

    # Weave the rest in to a fixed point: a plugin lands once its ``after`` anchor
    # is present in ``steps`` — a built-in id (the common case), an ``AT_START``
    # step already placed, or another plugin placed this round. This resolves anchor
    # chains regardless of discovery order; a plugin whose anchor never appears is
    # appended (below).
    plugin_after = {p.id: p.after for p in plugins}

    def _insert_after(anchor: str, step: Step) -> None:
        # Insert right after ``anchor``, but past any sibling plugin steps already
        # anchored on the same id, so plugins sharing an anchor keep input order.
        idx = next(i for i, s in enumerate(steps) if s.id == anchor)
        pos = idx + 1
        while pos < len(steps) and plugin_after.get(steps[pos].id) == anchor:
            pos += 1
        steps.insert(pos, step)

    placed_ids = {s.id for s in steps}
    remaining = [p for p in plugins if p.after != AT_START]  # AT_START already placed
    progress = True
    while remaining and progress:
        progress = False
        deferred: list[StepPlugin] = []
        for plugin in remaining:
            step = Step(plugin.id, plugin.label, by_step.get(plugin.id, []))
            if plugin.after is None:
                steps.append(step)
            elif plugin.after in placed_ids:
                _insert_after(plugin.after, step)
            else:
                deferred.append(plugin)  # anchor not placed yet — retry next pass
                continue
            placed_ids.add(plugin.id)
            progress = True
        remaining = deferred
    for plugin in remaining:  # unresolved anchor (typo / missing) → append at end
        steps.append(Step(plugin.id, plugin.label, by_step.get(plugin.id, [])))
    return steps


def build_model_choices(solver: Any) -> list[ModelChoice]:
    """Required (locked) + optional (add/remove) models from the catalog."""
    return [
        ModelChoice(name=e.name, label=e.label, required=e.required)
        for e in tools.model_catalog(solver)
    ]
