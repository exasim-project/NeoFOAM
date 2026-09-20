# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Wizard step model + model-selection choices (headless, pure)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from neofoam.mcp import tools
from neofoam.turbulence import momentumTransportModel
from neofoam.ui.forms import FormEntry, exclusive_model_families, humanize, js_identifier
from neofoam.ui.plugins import AT_START, StepPlugin

__all__ = [
    "Step",
    "ModelChoice",
    "ModelFamily",
    "build_steps",
    "build_model_choices",
    "build_model_families",
    "choice_key",
    "select_model_state",
    "selection_key",
    "turbulence_form_state",
    "loaded_turbulence_model",
]

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


@dataclass(frozen=True)
class ModelFamily:
    """A required family of alternatives — exactly one member runs per case."""

    name: str
    label: str
    members: list[ModelChoice]


def build_steps(
    solver: Any,
    entries: list[FormEntry],
    plugins: Sequence[StepPlugin] = (),
) -> list[Step]:
    """The built-in steps in fixed order, with any ``plugins`` woven in.

    Membership follows ``FormEntry.step`` (``models``/``schemes``/``bcs``/``initial``);
    ``geometry``/``sweep``/``review`` are form-less bespoke panels. Each plugin adds
    one extra step, slotted after its ``after`` anchor (an existing step id) or
    appended when the anchor is absent; ``plugins`` keep their discovery order
    among those sharing an anchor. A plugin's form-entry keys follow
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


def build_model_families(solver: Any) -> list[ModelFamily]:
    """The pick-one model families (:func:`neofoam.ui.forms.exclusive_model_families`).

    Their members are *alternatives*, not a set: the wizard renders one choice per
    family instead of listing every member as included. A required family with a single
    member offers no choice and is not here — it stays a locked "included" model.
    """
    by_name = {c.name: c for c in build_model_choices(solver)}
    return [
        ModelFamily(name=name, label=humanize(name), members=[by_name[m] for m in members])
        for name, members in exclusive_model_families(solver).items()
    ]


def selection_key(model: str) -> str:
    """The wizard state var (``sel_<model>``) holding whether ``model`` is selected.

    Use it wherever that switch is read, written or bound in a template, so a model
    name that is no JS identifier still gives a valid Vue expression, e.g.
    ``state[selection_key("kEpsilon")]``.
    """
    return js_identifier(f"sel_{model}")


def choice_key(family: str) -> str:
    """The wizard state var (``choice_<family>``) holding a family's selected member.

    The pick-one companion of :func:`selection_key`, e.g.
    ``state[choice_key("momentumTransportModel")]``.
    """
    return js_identifier(f"choice_{family}")


def select_model_state(families: list[ModelFamily], name: str) -> dict[str, Any]:
    """The wizard-state updates that select model ``name``.

    ``sel_<name>`` on, plus — when ``name`` is one alternative of a family — its
    siblings off and the family's ``choice_<family>`` moved to it, so a pick-one family
    can never end up with two members selected. Pure, so every route that selects a
    model (the Models step's radio group, the AI/``load_case`` fill auto-selecting the
    models it filled configs for) applies exactly the same rule.
    """
    updates: dict[str, Any] = {selection_key(name): True}
    for family in families:
        if any(c.name == name for c in family.members):
            updates[choice_key(family.name)] = name
            updates.update({selection_key(c.name): c.name == name for c in family.members})
    return updates


def turbulence_form_state(entries: list[FormEntry], name: str) -> dict[str, Any]:
    """The wizard-state update that makes ``turbulenceProperties`` select model ``name``.

    ``turbulenceProperties`` is shared by every momentum-transport model, so no single
    member owns (or gates) its form — the choice has to be written *into* it:
    ``simulationType`` is the family ``name`` was registered as, and a ``RAS``/``LES``
    model is named in the sub-dictionary of that family, next to that block's schema
    defaults (an unset boolean renders as an indeterminate checkbox). Empty when ``name``
    is not a momentum-transport model or the solver has no ``turbulenceProperties``
    form, so it can be applied to any model selection::

        state.update(turbulence_form_state(entries, "kOmegaSST"))
    """
    family = momentumTransportModel.family_of(name)
    entry = next((e for e in entries if e.cls_name == "TurbulencePropertiesConfig"), None)
    if family is None or entry is None:
        return {}
    data: dict[str, Any] = {"simulationType": family}
    if family != "laminar":
        block = entry.schema["properties"][family]["properties"]
        defaults = {key: node["default"] for key, node in block.items() if "default" in node}
        data[family] = {f"{family}Model": name, **defaults}
    return {entry.state_key: data}


def loaded_turbulence_model(entries: list[FormEntry], state: Any) -> str | None:
    """The momentum-transport model the ``turbulenceProperties`` form names, if any.

    The inverse of :func:`turbulence_form_state`, for a form filled from a case on
    disk: no model owns that config, so loading it selects none and the choice has to
    be read back out of the data::

        model = loaded_turbulence_model(entries, state)
    """
    entry = next((e for e in entries if e.cls_name == "TurbulencePropertiesConfig"), None)
    if entry is None:
        return None
    data = state[entry.state_key] or {}  # a form nothing has filled yet
    family = data.get("simulationType")
    if family == "laminar":
        return "laminar"
    model: str | None = (data.get(family) or {}).get(f"{family}Model")
    return model
