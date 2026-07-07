# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
"""Marimo UI to fill an ``incompressibleFluid`` case by hand.

Run with::

    marimo edit test/agent/hotRoom/case_ui.py

A sidebar lists every config the solver may consume
(:func:`configurations(incompressibleFluid)`); selecting one renders its
JSON-schema form next to it (via the ``json_schema_widget`` anywidget that
wraps ``@rjsf``). Edit the forms, hit each form's *Submit*, then *Save case*
to write the merged OpenFOAM case into this directory — the same
``write_configs`` path ``run_fill.py`` uses, just driven by hand instead of an
agent. No AI agent is involved here; that is a separate step.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    from json_schema_widget import Widget

    from neofoam.framework.solver.configurations import configurations
    from neofoam.io import write_configs
    from neofoam.solver.incompressibleFluid.incompressibleFluid import (
        incompressibleFluid,
    )

    try:
        TARGET = Path(__file__).resolve().parent
    except NameError:  # marimo cell without __file__
        TARGET = Path.cwd()
    return TARGET, Widget, configurations, incompressibleFluid, mo, write_configs


@app.cell
def _(mo):
    mo.md("""
    # incompressibleFluid case builder

    Pick a config on the left, fill its form on the right, and click the
    form's **Submit** button to push your edits back to Python. When every
    config you need is submitted, click **Save case** at the bottom to write
    the OpenFOAM case into this folder.

    > Edits only reach Python on **Submit** — a form you never submit keeps
    > its prefilled defaults.
    """)
    return


@app.cell
def _(Widget, configurations, incompressibleFluid, mo):
    cfgs = configurations(incompressibleFluid)

    def _default(cls):
        """Prefill the form with the config's defaults (no validation)."""
        try:
            return cls.model_construct().model_dump(by_alias=True, exclude_none=True)
        except Exception:
            return {}

    def _ui_schema(schema):
        """Tidy the @rjsf rendering of pydantic discriminated unions.

        The fvSchemes / fvSolution configs are deep trees of *discriminated
        unions* (pydantic emits ``oneOf`` + a ``discriminator`` whose branches
        each carry a ``type`` ``const``). Out of the box @rjsf renders every
        such field three times over: the union dropdown, a duplicate heading
        for the selected branch, and an editable text box for the ``const``
        ``type``. This uiSchema hides the ``const`` discriminator
        (``ui:widget: hidden``) — the value still round-trips so
        ``model_validate`` can pick the branch, it just isn't shown. The
        duplicate branch heading is removed in the widget itself (a custom
        ``TitleFieldTemplate`` dropping union-branch titles), which keeps the
        field/key labels visible.
        """
        defs = schema.get("$defs", {})

        def resolve(node):
            ref = node.get("$ref")
            if ref:
                return defs.get(ref.split("/")[-1], {})
            return node

        def for_node(node, seen):
            node = resolve(node)
            key = node.get("title")
            if key is not None:
                if key in seen:
                    return {}
                seen = seen | {key}
            ui = {}
            union = node.get("oneOf") or node.get("anyOf")
            if union:
                for branch in union:
                    for k, v in for_object(resolve(branch), seen).items():
                        ui.setdefault(k, v)
                return ui
            if node.get("type") == "object":
                ui.update(for_object(node, seen))
            return ui

        def for_object(obj, seen):
            ui = {}
            for name, prop in obj.get("properties", {}).items():
                pr = resolve(prop)
                if "const" in pr:
                    ui[name] = {"ui:widget": "hidden"}
                    continue
                sub = for_node(prop, seen)
                if sub:
                    ui[name] = sub
            return ui

        return for_node(schema, frozenset())

    # One cached widget per config so each form keeps its own edits when the
    # sidebar selection changes. The MUI theme is used because it ships its
    # styling via emotion (CSS-in-JS), so the form renders correctly inside
    # marimo without depending on an externally-applied Bootstrap stylesheet.
    widgets = {
        cls.__name__: mo.ui.anywidget(
            Widget(
                schema=cls.model_json_schema(),
                value=_default(cls),
                ui_schema=_ui_schema(cls.model_json_schema()),
                title=cls.__name__,
                theme="mui",
            )
        )
        for cls in cfgs
    }
    return cfgs, widgets


@app.cell
def _(cfgs, mo):
    selector = mo.ui.radio(options=cfgs.names, value=cfgs.names[0], label="**Configs**")
    return (selector,)


@app.cell
def _(mo, selector, widgets):
    mo.hstack(
        [selector, widgets[selector.value]],
        widths=[1, 4],
        align="start",
        gap=2,
    )
    return


@app.cell
def _(mo):
    save_btn = mo.ui.run_button(label="Save case")
    save_btn
    return (save_btn,)


@app.cell
def _(TARGET, cfgs, mo, save_btn, write_configs, widgets):
    mo.stop(
        not save_btn.value,
        mo.md("_Click **Save case** to write the submitted configs to disk._"),
    )

    instances = []
    errors: dict[str, str] = {}
    for cls in cfgs:
        data = widgets[cls.__name__].form_data
        if not data:
            continue
        try:
            instances.append(cls.model_validate(data))
        except Exception as exc:  # noqa: BLE001 - surfaced in the report
            errors[cls.__name__] = str(exc)

    report = write_configs(instances, case_dir=TARGET) if instances else {}

    lines = [f"### Saved to `{TARGET}`", ""]
    if report:
        for file, contribs in report.items():
            lines.append(f"- `{file}` ← {', '.join(contribs)}")
    else:
        lines.append("_Nothing valid to save — submit at least one form first._")
    if errors:
        lines += ["", "### Validation errors"]
        for name, msg in errors.items():
            lines.append(f"- **{name}**: {msg}")

    mo.md("\n".join(lines))
    return


if __name__ == "__main__":
    app.run()
