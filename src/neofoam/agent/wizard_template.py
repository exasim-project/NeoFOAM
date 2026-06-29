# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Packaged marimo *case wizard* notebook template + a writer to scaffold it.

``neofoam agent wizard <dir>`` writes :data:`NOTEBOOK_TEMPLATE` to
``<dir>/case_wizard.py``; the notebook is self-contained (imports its helpers
from :mod:`neofoam`) and operates on the directory it is saved in, so running
``marimo edit <dir>/case_wizard.py`` fills that case. Embedded as a string so it
ships reliably in the wheel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

__all__ = ["NOTEBOOK_TEMPLATE", "write_wizard_notebook"]


NOTEBOOK_TEMPLATE: str = r'''# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
"""Marimo wizard to fill an ``incompressibleFluid`` case as a guided journey.

Run with::

    marimo edit test/agent/hotRoom/case_wizard.py

Unlike ``case_ui.py`` (a flat sidebar of every config), this notebook lays the
same configs out as a two-step **user journey** in tabs:

* **Models** — the physics *Models* (controlDict, transport, turbulence,
  Boussinesq).
* **Schemes** — the numerical *Schemes* / linear solvers (fvSchemes /
  fvSolution); what they need depends on the models chosen.
* **BCs** — each field's ``boundaryField`` (the boundary conditions), edited on
  its own so the busiest part of case setup has a tab to itself.
* **Initial values** — each field's ``internalField``.

Each form is rendered from its pydantic JSON schema via the ``json_schema_widget``
anywidget (``@rjsf``). Edit a form, hit its *Submit*, then *Save case* to write
the merged OpenFOAM case into this directory via the same ``write_configs`` path
``run_fill.py`` uses.

An **AI chat** sits above the tabs: describe the case in natural language and a
pydantic-ai agent (``neofoam.agent.case_fill.build_case_agent``) fills the forms
and writes the case to disk. Requires ``ANTHROPIC_API_KEY`` in the environment;
without it the chat reports the error and the manual wizard still works.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    from json_schema_widget import Widget

    from neofoam.agent.case_forms import (
        INPUT_KEYS,
        field_name,
        is_scheme_config,
        merge_field_config,
        split_field_dump,
    )
    from neofoam.framework.solver.configurations import (
        configurations,
        model_catalog,
    )
    from neofoam.io import default_values, rjsf_uischema, write_configs, slice_schema
    from neofoam.solver.incompressibleFluid.incompressibleFluid import (
        incompressibleFluid,
    )

    try:
        TARGET = Path(__file__).resolve().parent
    except NameError:  # marimo cell without __file__
        TARGET = Path.cwd()
    return (
        INPUT_KEYS,
        TARGET,
        Widget,
        configurations,
        default_values,
        field_name,
        incompressibleFluid,
        is_scheme_config,
        merge_field_config,
        mo,
        rjsf_uischema,
        model_catalog,
        write_configs,
        slice_schema,
        split_field_dump,
    )


@app.cell
def _(mo):
    mo.md("""
    # Case wizard

    0. **Chat** — describe the case below and the AI fills the tabs and writes the
       case to disk. Then review/edit and **Save case** to re-write after edits.
    1. **Setup** — required (core) models are always on; add the optional ones
       you need (e.g. buoyancy). Unselected optional models are hidden everywhere
       below and left out of the saved case.
    2. **Models** — fill the physics models' configuration.
    3. **Schemes** — the numerical schemes / linear solvers (what they need
       depends on the models chosen).
    4. **BCs** — give every field its boundary conditions.
    5. **Initial values** — set each field's `internalField`.

    Expand a section, fill its form, and click that form's **Submit** to push the
    edits back to Python. When you are done, click **Save case** at the bottom to
    write the OpenFOAM case into this folder.

    > Edits only reach Python on **Submit** — a form you never submit keeps its
    > prefilled defaults (and a field you never touch at all is skipped). The AI
    > chat sets the values for you (no Submit needed for what it fills).
    """)
    return


@app.cell
def _(
    INPUT_KEYS,
    Widget,
    configurations,
    default_values,
    field_name,
    incompressibleFluid,
    is_scheme_config,
    mo,
    rjsf_uischema,
    slice_schema,
    model_catalog,
):
    cfgs = configurations(incompressibleFluid)

    # Every model the solver consumes, split into required (core; always on,
    # locked) and optional (toggleable). Each owns dict configs + 0/ fields —
    # derived from the solver's types, no hardcoding. The Setup tab locks the
    # required ones and gates the optional ones.
    _catalog = model_catalog(incompressibleFluid)
    required_models = [_e for _e in _catalog if _e.required]
    optional_models = [_e for _e in _catalog if not _e.required]

    def _make(schema, value, title):
        # Build the raw anywidget once and return both it and the marimo wrapper.
        # The AI chat mutates ``raw.form_data`` to display filled values; the
        # wrapper (used for layout) and the Save cell see the same trait.
        # MUI theme: its emotion (CSS-in-JS) styling survives marimo's shadow DOM.
        raw = Widget(
            schema=schema,
            value=value,
            ui_schema=rjsf_uischema(schema),
            title=title,
            theme="mui",
        )
        return mo.ui.anywidget(raw), raw

    # Dict configs (models + schemes) — one full-schema form each, keyed by name.
    dict_widgets = {}
    raw_dict_widgets = {}
    for _c in cfgs.dicts:
        _wrap, _raw = _make(_c.model_json_schema(), default_values(_c), _c.__name__)
        dict_widgets[_c.__name__] = _wrap
        raw_dict_widgets[_c.__name__] = _raw

    # Field configs — split each into an *input* slice (internalField/dimensions)
    # for the Initial-values tab and a *BCs* slice (boundaryField) for the BCs
    # tab. They are re-merged on save into one ``0/<field>`` config.
    input_widgets = {}
    bc_widgets = {}
    raw_input_widgets = {}
    raw_bc_widgets = {}
    field_label = {}
    for _cls in cfgs.fields:
        _name = _cls.__name__
        _fld = field_name(_cls)
        field_label[_name] = _fld
        _full = _cls.model_json_schema()
        _default_val = default_values(_cls)

        input_widgets[_name], raw_input_widgets[_name] = _make(
            slice_schema(_full, INPUT_KEYS),
            {k: v for k, v in _default_val.items() if k in INPUT_KEYS},
            f"{_fld} — initial value",
        )
        bc_widgets[_name], raw_bc_widgets[_name] = _make(
            slice_schema(_full, {"boundaryField"}),
            {"boundaryField": _default_val.get("boundaryField", {})},
            f"{_fld} — boundary conditions",
        )

    # Order the Setup tab: models/control first, then per-field inputs, then the
    # numerical schemes (fvSchemes/fvSolution live under ``system/fv*``).
    setup_models = [c.__name__ for c in cfgs.dicts if not is_scheme_config(c)]
    setup_schemes = [c.__name__ for c in cfgs.dicts if is_scheme_config(c)]
    field_names = [c.__name__ for c in cfgs.fields]
    return (
        bc_widgets,
        cfgs,
        dict_widgets,
        field_label,
        field_names,
        input_widgets,
        raw_bc_widgets,
        raw_dict_widgets,
        raw_input_widgets,
        optional_models,
        required_models,
        setup_models,
        setup_schemes,
    )


@app.cell
def _():
    # Build the pydantic-ai agent once. Its ``output_type`` is the aggregate
    # ``CaseSpec`` (one Optional field per config) — the same agent ``run_fill.py``
    # uses. Construction is wrapped so a missing ``ANTHROPIC_API_KEY`` doesn't stop
    # the notebook loading; the chat surfaces the error instead.
    from neofoam.agent.case_fill import build_case_agent, case_spec_to_configs

    MODEL_NAME = "claude-haiku-4-5"  # one-line swap to a sonnet/opus model
    try:
        agent = build_case_agent(model_name=MODEL_NAME)
        agent_error = None
    except Exception as exc:  # noqa: BLE001 - surfaced in the chat panel
        agent = None
        agent_error = str(exc)
    return agent, agent_error, case_spec_to_configs


@app.cell
def _(
    TARGET,
    agent,
    agent_error,
    case_spec_to_configs,
    field_names,
    get_sel,
    mo,
    optional_models,
    raw_bc_widgets,
    raw_dict_widgets,
    raw_input_widgets,
    set_sel,
    write_configs,
    split_field_dump,
):
    _history: list = []  # pydantic-ai message history → multi-turn refinement
    _field_set = set(field_names)

    async def _fill_model(messages, config):
        """Chat callback: prompt the agent, fill the forms, write the case.

        Async because marimo runs the chat callback inside a live event loop,
        where pydantic-ai's ``run_sync`` would raise "event loop is already
        running"; ``await agent.run(...)`` is the loop-friendly entry point.
        """
        if agent is None:
            return mo.md(
                "**AI agent unavailable.** Export `ANTHROPIC_API_KEY` and restart "
                f"the notebook.\n\n> {agent_error}"
            )
        try:
            result = await agent.run(messages[-1].content, message_history=_history)
            _history[:] = result.all_messages()
            # FieldValue serialises BC/internalField values to OpenFOAM literals
            # at write time, so the configs go straight to the widgets + disk.
            configs = case_spec_to_configs(result.output)

            # Sync the Setup selection with what the agent supplied: any optional
            # model it produced configs/fields for becomes "selected", so it shows
            # under Selected, its forms unhide, and a later manual Save keeps it.
            # (Without this the AI's choice and the wizard's UI go out of sync —
            # the agent fills Boussinesq but the Setup tab never reflects it.)
            _names = {type(c).__name__ for c in configs}
            _active_opt = {
                e.name
                for e in optional_models
                if any(c.__name__ in _names for c in (*e.dicts, *e.fields))
            }
            if _active_opt:
                set_sel(get_sel() | _active_opt)

            filled = []
            for cfg in configs:
                name = type(cfg).__name__
                dump = cfg.model_dump(by_alias=True, exclude_none=True)
                if name in _field_set:
                    # Push the field's two halves into their split widgets.
                    input_half, bc_half = split_field_dump(dump)
                    if name in raw_input_widgets:
                        raw_input_widgets[name].form_data = input_half
                    if name in raw_bc_widgets:
                        raw_bc_widgets[name].form_data = bc_half
                    filled.append(name)
                elif name in raw_dict_widgets:
                    raw_dict_widgets[name].form_data = dump
                    filled.append(name)

            # Auto-save: write the validated agent instances straight to disk.
            report = write_configs(configs, case_dir=TARGET) if configs else {}

            lines = [
                "**Filled:** " + (", ".join(sorted(filled)) if filled else "_nothing_")
            ]
            if _active_opt:
                _labels = [e.label for e in optional_models if e.name in _active_opt]
                lines.append("**Selected models:** " + ", ".join(sorted(_labels)))
            if report:
                lines += ["", f"**Wrote to** `{TARGET}`:"]
                lines += [f"- `{f}` ← {', '.join(c)}" for f, c in report.items()]
            lines += [
                "",
                "Review/edit the tabs below; click **Save case** after manual edits.",
            ]
            return mo.md("\n".join(lines))
        except Exception as exc:  # noqa: BLE001 - surfaced in the chat panel
            return mo.md(f"**Fill failed:** {exc}")

    chat = mo.ui.chat(
        _fill_model,
        prompts=[
            "Lid-driven cavity, laminar, kEpsilon; top patch movingWall, the "
            "other patches fixedWalls",
            "Buoyant hot room with the Boussinesq model and kEpsilon; patches "
            "top bottom left right front back",
        ],
    )
    chat
    return


@app.cell
def _(mo):
    # Selected optional-model names. Plain state so the Add/Remove buttons can
    # mutate it; the buttons live in cells that read only the *stable* Available
    # table (below), never this state at definition time — that keeps them
    # reliable (the earlier per-row remove buttons failed because their cell both
    # read and set the state, tripping marimo's loop guard).
    get_sel, set_sel = mo.state(set())
    return get_sel, set_sel


@app.cell
def _(mo, optional_models):
    # Transfer list — Available: a selectable ``mo.ui.table`` of the *optional*
    # models (scales: search + pagination). Required (core) models aren't here —
    # they're always on and shown locked in the Selected table. Stable: depends
    # only on ``optional_models``, so the Add/Remove buttons stay live.
    catalog_table = mo.ui.table(
        [{"Model": e.label} for e in optional_models],
        selection="multi",
        label="Available (optional)",
        pagination=False,
    )
    return (catalog_table,)


@app.cell
def _(catalog_table, get_sel, mo, optional_models, set_sel):
    # Add the rows currently selected in the Available table to the selection.
    _l2n = {e.label: e.name for e in optional_models}

    def _add(_v):
        set_sel(get_sel() | {_l2n[r["Model"]] for r in catalog_table.value})

    add_btn = mo.ui.button(label="Add →", on_change=_add)
    return (add_btn,)


@app.cell
def _(catalog_table, get_sel, mo, optional_models, set_sel):
    # Remove the rows selected in the Available table from the selection.
    _l2n = {e.label: e.name for e in optional_models}

    def _remove(_v):
        set_sel(get_sel() - {_l2n[r["Model"]] for r in catalog_table.value})

    remove_btn = mo.ui.button(label="← Remove", on_change=_remove)
    return (remove_btn,)


@app.cell
def _(get_sel, mo, optional_models, required_models):
    # Transfer list — Selected: required (core) models are always present and
    # marked locked; chosen optional models follow.
    _n2l = {e.name: e.label for e in optional_models}
    selected_table = mo.ui.table(
        [{"Model": e.label, "": "required"} for e in required_models]
        + [{"Model": _n2l[_n], "": ""} for _n in sorted(get_sel())],
        selection=None,
        label="Selected",
        pagination=False,
    )
    return (selected_table,)


@app.cell
def _(
    add_btn,
    bc_widgets,
    catalog_table,
    dict_widgets,
    field_label,
    field_names,
    get_sel,
    input_widgets,
    mo,
    optional_models,
    remove_btn,
    selected_table,
    setup_models,
    setup_schemes,
):
    # Setup tab: required (core) models are always on; pick the optional ones. An
    # unselected optional model has its configs + fields hidden across the other
    # tabs (and skipped on save). Required models are never hidden.
    _selected_types = set(get_sel())
    _hidden_dicts = {
        _c.__name__
        for _e in optional_models
        if _e.name not in _selected_types
        for _c in _e.dicts
    }
    _hidden_fields = {
        _c.__name__
        for _e in optional_models
        if _e.name not in _selected_types
        for _c in _e.fields
    }

    setup_panel = mo.vstack(
        [
            mo.md(
                "### Models\n"
                "Required (core) models are always on (locked, shown under "
                "**Selected**). For **optional** models, select rows under "
                "**Available** then **Add →** (or **← Remove**); unselected "
                "optional models stay hidden and are left out of the saved case."
            ),
            mo.hstack(
                [
                    mo.vstack([catalog_table, add_btn]),
                    mo.vstack([selected_table, remove_btn]),
                ],
                justify="start",
                gap=3,
            ),
        ]
    )

    models_items = {
        _name: dict_widgets[_name]
        for _name in setup_models
        if _name not in _hidden_dicts
    }
    schemes_items = {
        _name: dict_widgets[_name]
        for _name in setup_schemes
        if _name not in _hidden_dicts
    }
    bcs_items = {
        f"{field_label[_name]} — boundary conditions": bc_widgets[_name]
        for _name in field_names
        if _name not in _hidden_fields
    }
    initial_items = {
        f"{field_label[_name]} — initial value": input_widgets[_name]
        for _name in field_names
        if _name not in _hidden_fields
    }

    wizard = mo.ui.tabs(
        {
            "Setup": setup_panel,
            "Models": mo.accordion(models_items),
            "Schemes": mo.accordion(schemes_items),
            "BCs": mo.accordion(bcs_items),
            "Initial values": mo.accordion(initial_items),
        }
    )
    wizard
    return


@app.cell
def _(mo):
    save_btn = mo.ui.run_button(label="Save case")
    save_btn
    return (save_btn,)


@app.cell
def _(
    TARGET,
    bc_widgets,
    cfgs,
    dict_widgets,
    field_names,
    get_sel,
    input_widgets,
    merge_field_config,
    mo,
    optional_models,
    save_btn,
    write_configs,
):
    mo.stop(
        not save_btn.value,
        mo.md("_Click **Save case** to write the submitted configs to disk._"),
    )

    # Skip configs/fields owned by an optional model that isn't selected
    # (required/core models are never skipped).
    _selected_types = set(get_sel())
    _skip = {
        _c.__name__
        for _e in optional_models
        if _e.name not in _selected_types
        for _c in (*_e.dicts, *_e.fields)
    }

    _field_set = set(field_names)
    _instances = []
    _errors: dict[str, str] = {}
    for _cls in cfgs:
        _name = _cls.__name__
        if _name in _skip:
            continue
        try:
            if _name in _field_set:
                # A field with neither half submitted is skipped; otherwise its
                # two halves are merged over the config defaults (see
                # ``merge_field_config``), so an unsubmitted half falls back.
                _in_fd = input_widgets[_name].form_data
                _bc_fd = bc_widgets[_name].form_data
                if not _in_fd and not _bc_fd:
                    continue
                _instances.append(merge_field_config(_cls, _in_fd, _bc_fd))
            else:
                _data = dict_widgets[_name].form_data
                if not _data:
                    continue
                _instances.append(_cls.model_validate(_data))
        except Exception as exc:  # noqa: BLE001 - surfaced in the report
            _errors[_name] = str(exc)

    _report = write_configs(_instances, case_dir=TARGET) if _instances else {}

    _lines = [f"### Saved to `{TARGET}`", ""]
    if _report:
        for _file, _contribs in _report.items():
            _lines.append(f"- `{_file}` ← {', '.join(_contribs)}")
    else:
        _lines.append("_Nothing valid to save — submit at least one form first._")
    if _errors:
        _lines += ["", "### Validation errors"]
        for _ename, _msg in _errors.items():
            _lines.append(f"- **{_ename}**: {_msg}")

    mo.md("\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
'''


def write_wizard_notebook(
    target_dir: Union[Path, str],
    *,
    filename: str = "case_wizard.py",
    force: bool = False,
) -> Path:
    """Write the case-wizard notebook into ``target_dir`` and return its path."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / filename
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists; pass force=True (CLI: --force) to overwrite"
        )
    path.write_text(NOTEBOOK_TEMPLATE)
    return path
