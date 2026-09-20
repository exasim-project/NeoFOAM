# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Trame case wizard: sidebar step nav + JSONForms panels + model-aware save.

The left drawer lists the wizard steps (``current_step``); the main area mounts every
step once, shown via ``v_show`` so navigation never unmounts a form and wipes its
state. Every form is rendered generically from its :class:`~neofoam.ui.forms.FormEntry`
JSON Schema — there is no per-case or per-config markup. The Models step opens with
the model-selection panel (``sel_<model>``): a toggle per optional model and a radio
group per pick-one family (``choice_<family>``, exactly one member selected). A config
panel owned by an unselected model is hidden and skipped on save. Save aggregates the
live form state
through :func:`neofoam.ui.case_spec.state_to_case_spec` and writes via
:func:`neofoam.mcp.tools.save_case`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from neofoam.mcp import tools
from neofoam.mcp.registry import resolve_solver
from neofoam.ui import case_spec as cs
from neofoam.ui import jsonforms_module
from neofoam.ui._paths import _resolve_target
from neofoam.ui.agent_panel import build_agent_panel
from neofoam.ui.forms import (
    ADDER_TRANSLATIONS,
    FormEntry,
    build_forms,
    schema_key,
    uischema_key,
)
from neofoam.ui.geometry_panel import GeometryPanel
from neofoam.ui.plugins import StepContext, StepPlugin, discover_step_plugins
from neofoam.ui.review import FindingRow, findings_to_rows, save_error_rows
from neofoam.ui.scaffold import scaffold_runnable_case
from neofoam.ui.steps import (
    Step,
    build_model_choices,
    build_model_families,
    build_steps,
    choice_key,
    select_model_state,
    selection_key,
    turbulence_form_state,
)
from neofoam.ui.sweep_panel import SweepPanel

# Per-step presentation (icon + one-line caption under the step title). Purely
# cosmetic — steps themselves come from ``neofoam.ui.steps.build_steps``.
_STEP_ICONS = {
    "models": "mdi-atom",
    "geometry": "mdi-cube-outline",
    "bcs": "mdi-border-all-variant",
    "initial": "mdi-waves",
    "schemes": "mdi-function-variant",
    "sweep": "mdi-tune-variant",
    "review": "mdi-clipboard-check-outline",
}
_STEP_CAPTIONS = {
    "models": "Pick the optional physics models, then configure the model dictionaries.",
    "geometry": "Scan the boundary STLs, assign patch roles and author the mesh dicts.",
    "bcs": "Boundary conditions per field — patches are seeded by the geometry scan.",
    "initial": "Physical dimensions and initial internal value per field.",
    "schemes": "Discretisation schemes and linear solvers — the defaults are sensible.",
    "sweep": "Sweep any config over named variants and export a Snakemake workflow.",
    "review": "Save the case, review the validation findings and run it.",
}

# One coherent look for every (schema-generated) widget: an app theme plus global
# component defaults, so the generic forms need no per-widget styling.
_VUETIFY_CONFIG = {
    "theme": {
        "defaultTheme": "neofoam",
        "themes": {
            "neofoam": {
                "dark": False,
                "colors": {
                    "primary": "#1F5FBF",
                    "secondary": "#00838F",
                    "background": "#F4F6FB",
                    "surface": "#FFFFFF",
                    "surface-variant": "#E9EDF5",
                    # Vuetify's stock surface-variant is dark, so its stock
                    # on-surface-variant is near-white. Overriding only the
                    # background left every `bg-surface-variant` element (the
                    # assistant's chat replies) white-on-light, i.e. invisible.
                    "on-surface-variant": "#1F2A3C",
                    "error": "#C62828",
                    "warning": "#E65100",
                    "success": "#2E7D32",
                    "info": "#0277BD",
                },
            },
        },
    },
    "defaults": {
        "VTextField": {"density": "compact", "variant": "outlined", "color": "primary"},
        "VSelect": {"density": "compact", "variant": "outlined", "color": "primary"},
        "VCombobox": {"density": "compact", "variant": "outlined", "color": "primary"},
        "VNumberInput": {"density": "compact", "variant": "outlined", "color": "primary"},
        "VSwitch": {"density": "compact", "color": "primary", "hideDetails": True},
        "VBtn": {"rounded": "lg"},
        "VCard": {"rounded": "lg"},
        "VAlert": {"density": "compact", "rounded": "lg"},
        "VExpansionPanels": {"multiple": True, "variant": "accordion"},
        "VTooltip": {"location": "bottom"},
    },
}

# Light global polish on top of the theme (panel borders, calmer form spacing).
_CSS = """
.v-expansion-panel { border: 1px solid rgba(31, 95, 191, 0.12); }
.v-expansion-panel-title { font-weight: 500; min-height: 44px; }
.v-expansion-panel--active > .v-expansion-panel-title { color: rgb(31, 95, 191); }
.nf-step-title { letter-spacing: -0.3px; }
.v-navigation-drawer .v-list-item-title { font-weight: 500; }
/* Phone/tablet (below Vuetify's md breakpoint, where both drawers are overlays). */
@media (max-width: 959.98px) {
  .v-navigation-drawer { max-width: calc(100vw - 56px); }
  .v-main .v-expansion-panel-text__wrapper { padding: 8px 8px 16px; }
  .v-main .v-btn:not(.v-btn--icon), .v-app-bar .v-btn:not(.v-btn--icon) { min-height: 40px; }
}
@media (max-width: 599.98px) {
  .v-app-bar .v-toolbar-title { display: none; }
}
"""

# Below Vuetify's md breakpoint (960 px) both drawers are closed-by-default overlays
# with their own open flags, so a phone never inherits the desktop's open drawers.
_MOBILE = "$vuetify.display.smAndDown"


def _responsive_open(desktop_flag: str, mobile_flag: str) -> dict[str, Any]:
    """Drawer props: permanent + ``desktop_flag`` on desktop, overlay + ``mobile_flag`` below."""
    return {
        "permanent": (f"!{_MOBILE}",),
        "temporary": (_MOBILE,),
        "model_value": (f"{_MOBILE} ? {mobile_flag} : {desktop_flag}",),
        "update_modelValue": f"{_MOBILE} ? ({mobile_flag} = $event) : ({desktop_flag} = $event)",
    }


def _toggle(desktop_flag: str, mobile_flag: str) -> str:
    """JS toggling whichever open flag the current display width uses."""
    return f"{_MOBILE} ? ({mobile_flag} = !{mobile_flag}) : ({desktop_flag} = !{desktop_flag})"


def build_app(
    server: Any = None,
    *,
    solver_name: str = "incompressibleFluid",
    plugins: list[StepPlugin] | None = None,
) -> Any:
    """Construct the trame Server, state, layout and Save controller. Returns it.

    ``plugins`` overrides step-plugin discovery (a test/embedding seam); by default
    the ``neofoam.ui.steps`` entry points are used. Contributed steps are woven into
    the sidebar and content area after the built-in steps are wired.
    """
    from trame.app import get_server  # type: ignore  # untyped optional 'ui' dep  # noqa: PLC0415
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout  # type: ignore  # noqa: PLC0415
    from trame.widgets import client, html  # type: ignore  # untyped deps  # noqa: PLC0415
    from trame.widgets import vuetify3 as v3  # noqa: PLC0415
    from trame_client.widgets.core import AbstractElement  # type: ignore  # noqa: PLC0415

    class JsonForms(AbstractElement):  # type: ignore[misc]  # untyped base
        """The bundled ``<json-forms>`` client component (JSONForms + Vuetify)."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__("json-forms", **kwargs)
            self._attr_names += ["schema", "uischema", "data", "translations"]
            self._event_names += ["change"]

    solver = resolve_solver(solver_name)
    entries: list[FormEntry] = build_forms(solver)
    step_plugins = discover_step_plugins(plugins)
    steps = build_steps(solver, entries, step_plugins)
    plugin_by_id = {p.id: p for p in step_plugins}
    choices = build_model_choices(solver)
    # A required family of alternatives is a choice (one member at a time); every other
    # required model is always on. `gated` is every model with a `sel_<name>` switch.
    families = build_model_families(solver)
    family_of = {c.name: f for f in families for c in f.members}
    required = [c for c in choices if c.required and c.name not in family_of]
    optional = [c for c in choices if not c.required]
    gated = [*optional, *(c for f in families for c in f.members)]
    by_key = {e.key: e for e in entries}
    label_of = {c.name: c.label for c in choices}

    server = get_server() if server is None else server
    state, ctrl = server.state, server.controller

    state.current_step = steps[0].id
    state.ai_panel = True  # right AI drawer open by default (foldable)
    state.ai_panel_mobile = False
    state.main_drawer_mobile = False
    state.target_dir = ""
    state.save_report = None
    state.scaffolded = []
    state.validation_ok = None
    state.findings = []
    for c in optional:
        state[selection_key(c.name)] = False
    # Each family starts on its first registered member — exactly one runs per case, so
    # "none selected" is not a valid state to save from.
    for family in families:
        state[choice_key(family.name)] = family.members[0].name
        for c in family.members:
            state[selection_key(c.name)] = c is family.members[0]
    state.form_translations = ADDER_TRANSLATIONS
    for entry in entries:
        state[schema_key(entry)] = entry.schema
        state[entry.state_key] = dict(entry.defaults)
        if entry.uischema is not None:
            state[uischema_key(entry)] = entry.uischema
    # turbulenceProperties follows the turbulence choice rather than starting empty;
    # a solver with no such family (VoF reads the file itself) starts laminar.
    state.update(turbulence_form_state(entries, "laminar"))
    for family in families:
        state.update(turbulence_form_state(entries, family.members[0].name))

    # ------------------------------------------------------------------ #
    # Controller                                                         #
    # ------------------------------------------------------------------ #

    def _validate_and_store(case_dir: Path) -> None:
        report = tools.validate_case(solver, str(case_dir))
        state.validation_ok = report.ok
        state.findings = [asdict(r) for r in findings_to_rows(report)]

    def _report_path_error(exc: ValueError) -> None:
        """A blank/relative path field is a UI problem, not a config problem."""
        state.validation_ok = False
        state.findings = [
            asdict(
                FindingRow(
                    level="error",
                    color="error",
                    file="Target directory",
                    message=str(exc),
                    fix="Type an absolute case directory in the toolbar field.",
                )
            )
        ]

    def select_model(name: str) -> None:
        """Select model ``name`` (deselecting its siblings when it is one alternative)."""
        state.update(select_model_state(families, name))
        state.update(turbulence_form_state(entries, name))

    def save_case() -> None:
        selected = {c.name for c in gated if state[selection_key(c.name)]}
        form_state = {e.key: dict(state[e.state_key]) for e in entries}
        state.current_step = "review"
        try:
            target = _resolve_target(state.target_dir, "target directory")
        except ValueError as exc:
            state.save_report = {"error": str(exc)}
            state.scaffolded = []
            _report_path_error(exc)
            return
        try:
            # state_to_case_spec merges/validates field halves and may itself raise.
            spec = cs.state_to_case_spec(entries, form_state, selected)
            result = tools.save_case(solver, spec, str(target))
            state.save_report = result.model_dump()
            # Make the saved case runnable, then validate it — inside the try, so a
            # scaffold/validate failure is reported instead of leaving Review saying
            # the case still has to be saved.
            state.scaffolded = [str(p) for p in scaffold_runnable_case(target)]
            _validate_and_store(target)
        except Exception as exc:  # noqa: BLE001 - a save failure must not crash the UI
            # Incomplete/invalid configs — surface each field instead of crashing.
            state.save_report = {"error": str(exc)}
            state.scaffolded = []
            state.validation_ok = False
            state.findings = [asdict(r) for r in save_error_rows(exc)]

    def revalidate() -> None:
        try:
            target = _resolve_target(state.target_dir, "target directory")
        except ValueError as exc:
            _report_path_error(exc)
            return
        try:
            _validate_and_store(target)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
            state.validation_ok = False
            state.findings = [asdict(r) for r in save_error_rows(exc)]

    ctrl.select_model = select_model
    ctrl.save_case = save_case
    ctrl.revalidate = revalidate
    ctrl.get_entries = lambda: entries
    ctrl.get_steps = lambda: steps

    geometry_panel = GeometryPanel(server, entries)
    build_agent_panel(server, entries, solver)
    sweep_panel = SweepPanel(server, entries, solver, solver_name)

    # Contributed steps (§ neofoam.ui.plugins): each seeds its own state +
    # controllers now, and draws its panel in the content loop below.
    ctx = StepContext(
        server=server,
        solver=solver,
        solver_name=solver_name,
        entries=entries,
        json_forms=JsonForms,
        v3=v3,
        html=html,
        client=client,
        sweep=sweep_panel,
        schema_key=schema_key,
    )
    for plugin in step_plugins:
        plugin.register(ctx)

    # ------------------------------------------------------------------ #
    # Layout                                                             #
    # ------------------------------------------------------------------ #

    def _step_header(step: Step) -> None:
        _p = plugin_by_id.get(step.id)
        caption = _p.caption if _p is not None else _STEP_CAPTIONS.get(step.id, "")
        html.Div(step.label, classes="text-h5 font-weight-bold nf-step-title")
        html.Div(
            caption,
            classes="text-body-2 text-medium-emphasis mb-5",
        )

    def _form_panel(entry: FormEntry) -> None:
        # Owned by a selectable model → hide unless selected; else always visible
        # (omit v-show entirely — a bare `v-show` with no expression won't compile).
        panel_kwargs = {}
        if entry.owner_model is not None:
            panel_kwargs["v_show"] = selection_key(entry.owner_model)
        form_kwargs = {}
        if entry.uischema is not None:
            form_kwargs["uischema"] = (uischema_key(entry),)
        with v3.VExpansionPanel(elevation=0, **panel_kwargs):
            with v3.VExpansionPanelTitle():
                html.Span(entry.title)
                if entry.owner_model is not None:
                    v3.VChip(
                        label_of[entry.owner_model],
                        size="x-small",
                        color="secondary",
                        variant="tonal",
                        classes="ml-2",
                    )
            with v3.VExpansionPanelText():
                JsonForms(
                    schema=(schema_key(entry),),
                    data=(entry.state_key,),
                    translations=("form_translations",),
                    change=f"{entry.state_key} = $event.data",
                    **form_kwargs,
                )

    def _model_selector() -> None:
        """Always-on models (locked) + one pick-one control per family + toggles."""
        with v3.VCard(variant="outlined", classes="mb-6"):
            with v3.VCardText():
                if required:
                    html.Div(
                        "Included models",
                        classes="text-overline text-medium-emphasis",
                    )
                    with html.Div(classes="d-flex flex-wrap ga-2 mb-4"):
                        for c in required:
                            v3.VChip(
                                c.label,
                                prepend_icon="mdi-lock",
                                variant="tonal",
                                color="primary",
                                size="small",
                            )
                # One member at a time: the radio group is the only way to select one,
                # so picking a member deselects its siblings server-side.
                for family in families:
                    html.Div(
                        family.label,
                        classes="text-overline text-medium-emphasis",
                    )
                    with v3.VRadioGroup(
                        model_value=(choice_key(family.name),),
                        update_modelValue=(ctrl.select_model, "[$event]"),
                        inline=True,
                        hide_details=True,
                        classes="mb-4",
                    ):
                        for c in family.members:
                            v3.VRadio(label=c.label, value=c.name)
                html.Div(
                    "Optional models",
                    classes="text-overline text-medium-emphasis",
                )
                for c in optional:
                    v3.VSwitch(v_model=(selection_key(c.name),), label=c.label, inset=True)

    def _review_panel() -> None:
        """validate_case findings + scaffolded runnable case."""
        with v3.VRow(align="center", classes="mb-2", no_gutters=True):
            v3.VSpacer()
            v3.VBtn(
                "Re-validate",
                click=ctrl.revalidate,
                variant="tonal",
                color="primary",
                prepend_icon="mdi-refresh",
            )
        # Prompt to save first.
        v3.VAlert(
            "Click 'Save case' to write the case, scaffold Allrun/Allclean and validate it.",
            type="info",
            variant="tonal",
            classes="mb-3",
            v_show="validation_ok === null",
        )
        # Overall verdict.
        v3.VAlert(
            text=(
                "validation_ok"
                " ? 'Case is valid — run it with ./Allrun'"
                " : (findings.length + ' issue(s) to fix before it will run')",
            ),
            type=("validation_ok ? 'success' : 'error'",),
            variant="tonal",
            classes="mb-3",
            v_show="validation_ok !== null",
        )
        # One alert per finding.
        with v3.VAlert(
            v_for="(f, i) in findings",
            key="i",
            type=("f.color",),
            variant="tonal",
            border="start",
            classes="mb-2",
        ):
            v3.VAlertTitle("{{ f.file }}")
            html.Div("{{ f.message }}")
            html.Div(
                "Fix → {{ f.fix }}",
                v_show="f.fix",
                classes="text-medium-emphasis mt-1",
            )
        # Scaffolded files.
        v3.VAlert(
            text=("'Scaffolded: ' + scaffolded.join(', ')",),
            type="success",
            variant="outlined",
            classes="mt-3",
            v_show="scaffolded.length",
        )

    def _ai_drawer() -> None:
        """Right-hand foldable AI chat drawer (multi-turn, fills the forms)."""
        with v3.VNavigationDrawer(
            location="right",
            width=400,
            **_responsive_open("ai_panel", "ai_panel_mobile"),
        ):
            with html.Div(classes="d-flex flex-column", style="height: 100%;"):
                with v3.VToolbar(title="AI assistant", density="compact", flat=True):
                    # The overlay leaves only a sliver of scrim to tap on a phone.
                    v3.VBtn(
                        icon="mdi-close",
                        click="ai_panel_mobile = false",
                        v_if=_MOBILE,
                    )
                # Scrolling transcript.
                with html.Div(classes="flex-grow-1 pa-3", style="overflow-y: auto;"):
                    # Empty-state hint + suggested prompts.
                    with html.Div(v_show="!chat_log.length"):
                        v3.VCardText(
                            "Describe your case and I'll fill the forms. Try:",
                            classes="text-medium-emphasis px-0",
                        )
                        with v3.VChip(
                            v_for="(p, i) in suggested_prompts",
                            key="i",
                            click=(ctrl.send_message, "[p]"),
                            disabled=("ai_busy",),
                            size="small",
                            variant="tonal",
                            color="secondary",
                            classes="mb-2",
                            style="height: auto; white-space: normal;",
                        ):
                            html.Span("{{ p }}", classes="py-1")
                    # Messages.
                    with v3.VSheet(
                        v_for="(m, i) in chat_log",
                        key="i",
                        rounded="lg",
                        classes="pa-3 mb-2",
                        color=("m.role === 'user' ? 'primary' : 'surface-variant'",),
                    ):
                        html.Div(
                            "{{ m.content }}",
                            style="white-space: pre-wrap; font-size: 0.9rem;",
                        )
                    v3.VProgressLinear(indeterminate=True, v_show="ai_busy", color="secondary")
                # Composer pinned to the bottom.
                with html.Div(classes="pa-3"):
                    v3.VTextField(
                        v_model=("chat_input",),
                        placeholder="Message the assistant…",
                        hide_details=True,
                        keydown_enter=(ctrl.send_message, "[]"),
                        disabled=("ai_busy",),
                    )
                    v3.VBtn(
                        "Send",
                        click=(ctrl.send_message, "[]"),
                        loading=("ai_busy",),
                        disabled=("!chat_input || ai_busy",),
                        color="secondary",
                        prepend_icon="mdi-send",
                        block=True,
                        classes="mt-2",
                    )

    with SinglePageWithDrawerLayout(server, vuetify_config=_VUETIFY_CONFIG) as layout:
        layout.title.set_text("NeoFOAM case wizard")
        client.Style(_CSS)

        # The layout's own v-model would fight the display-dependent model below.
        layout.drawer.v_model = None
        for prop, value in _responsive_open("main_drawer", "main_drawer_mobile").items():
            setattr(layout.drawer, prop, value)
        layout.icon.click = _toggle("main_drawer", "main_drawer_mobile")

        with layout.drawer:
            with v3.VList(nav=True, density="comfortable", color="primary"):
                v3.VListSubheader(solver_name)
                for step in steps:
                    _p = plugin_by_id.get(step.id)
                    v3.VListItem(
                        title=step.label,
                        prepend_icon=(
                            _p.icon
                            if _p is not None
                            else _STEP_ICONS.get(step.id, "mdi-circle-outline")
                        ),
                        rounded="lg",
                        active=(f"current_step === '{step.id}'",),
                        click=f"current_step = '{step.id}'; main_drawer_mobile = false",
                    )

        with layout.toolbar:
            v3.VSpacer()
            v3.VTextField(
                v_model=("target_dir",),
                label="Target directory",
                hide_details=True,
                prepend_inner_icon="mdi-folder-arrow-down-outline",
                style="max-width: 340px",
                v_if=f"!{_MOBILE}",
            )
            v3.VBtn(
                "Save case",
                click=ctrl.save_case,
                color="primary",
                variant="flat",
                prepend_icon="mdi-content-save-outline",
                classes="mx-3",
                # Without a target the case would land in the server's launch dir.
                disabled=("!target_dir.trim()",),
            )
            # Fold / unfold the AI assistant drawer.
            v3.VBtn(
                icon="mdi-robot-happy-outline",
                click=_toggle("ai_panel", "ai_panel_mobile"),
                variant="text",
            )
            # A phone toolbar has no room for the path: it gets a second row.
            with v3.Template(v_if=_MOBILE, v_slot_extension=True):
                v3.VTextField(
                    v_model=("target_dir",),
                    label="Target directory",
                    hide_details=True,
                    prepend_inner_icon="mdi-folder-arrow-down-outline",
                    classes="mx-3",
                )

        with layout.root:
            _ai_drawer()

        with layout.content, v3.VContainer(fluid=True, classes="pa-4 pa-md-6"):
            # One uniform loop: every step renders its header, its bespoke panel
            # (models / geometry / review) and its schema-generated form panels.
            for step in steps:
                with html.Div(v_show=f"current_step === '{step.id}'"):
                    _step_header(step)
                    if step.id == "models":
                        _model_selector()
                    elif step.id == "geometry":
                        geometry_panel.render(v3, html)
                    elif step.id == "bcs":
                        v3.VAlert(
                            "No patches scanned yet — run Scan in the Geometry step to seed"
                            " them, or add one by hand below.",
                            type="info",
                            variant="tonal",
                            classes="mb-4",
                            v_show="!geometry_patches.length",
                        )
                    elif step.id == "sweep":
                        sweep_panel.render(JsonForms)
                    elif step.id == "review":
                        _review_panel()
                    elif step.id in plugin_by_id:
                        plugin_by_id[step.id].render(ctx)
                    if step.entry_keys:
                        with v3.VExpansionPanels():
                            for key in step.entry_keys:
                                _form_panel(by_key[key])

    server.enable_module(jsonforms_module)
    return server
