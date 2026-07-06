# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Trame case wizard: sidebar step nav + JSONForms panels + model-aware save.

The left drawer lists the wizard steps (``current_step``); the main area mounts every
step once, shown via ``v_show`` so navigation never unmounts a form and wipes its
state. Every form is rendered generically from its :class:`~neofoam.ui.forms.FormEntry`
JSON Schema — there is no per-case or per-config markup. The Models step opens with
the model-selection panel (``sel_<model>``); a config panel owned by an unselected
optional model is hidden and skipped on save. Save aggregates the live form state
through :func:`neofoam.ui.case_spec.state_to_case_spec` and writes via
:func:`neofoam.mcp.tools.save_case`.
"""

from __future__ import annotations

import os
from typing import Any

from dataclasses import asdict
from pathlib import Path

from neofoam.mcp import tools
from neofoam.mcp.registry import resolve_solver
from neofoam.ui import case_spec as cs
from neofoam.ui import jsonforms_module
from neofoam.ui.agent_panel import build_agent_panel
from neofoam.ui.forms import (
    FormEntry,
    build_forms,
    patch_bc_schema,
    seed_boundary_field,
)
from neofoam.ui.geometry import (
    GeometrySpec,
    MeshSettings,
    PatchGeometry,
    PatchRole,
    discover_geometry,
    write_mesh_configs,
)
from neofoam.ui import review
from neofoam.ui.review import findings_to_rows
from neofoam.ui.scaffold import scaffold_runnable_case
from neofoam.ui.steps import Step, build_model_choices, build_steps
from neofoam.ui.sweep_panel import SweepPanel

_ROLE_NAMES = [r.value for r in PatchRole]

# Optional pre-fill for the geometry step's STL folder (manual-testing convenience).
_DEFAULT_STL_DIR = os.environ.get("NEOFOAM_WIZARD_STL_DIR", "")

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
"""


def _case_dir_for(stl_dir: str) -> str:
    """The case dir owning ``stl_dir`` (its ``constant/triSurface`` parent), else itself."""
    p = Path(stl_dir)
    if p.name == "triSurface" and p.parent.name == "constant":
        return str(p.parent.parent)
    return stl_dir


def _patch_row(patch: PatchGeometry) -> dict[str, Any]:
    """A ``PatchGeometry`` as an editable trame-state row."""
    ref = patch.refinement
    return {
        "name": patch.name,
        "stl": patch.stl,
        "role": patch.role.value,
        "is_snappy": patch.is_snappy_surface,
        "box_faces": list(patch.box_faces) if patch.box_faces else None,
        "faces": ", ".join(patch.box_faces) if patch.box_faces else "snappy surface",
        "refinement_str": f"{ref[0]} {ref[1]}" if ref else "",
    }


def _parse_pair(text: str) -> tuple[int, int] | None:
    """Parse a ``"min max"`` refinement string; ``None`` when unparsable."""
    parts = text.split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _parse_vec(text: str) -> tuple[float, float, float] | None:
    parts = text.split()
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def _spec_from_state(state: Any) -> GeometrySpec:
    """Rebuild a ``GeometrySpec`` from the (possibly user-edited) geometry state."""
    patches: list[PatchGeometry] = []
    for row in state.geometry_patches:
        refinement = None
        if row["is_snappy"]:
            refinement = _parse_pair(row.get("refinement_str", "")) or (1, 2)
        patches.append(
            PatchGeometry(
                name=row["name"],
                stl=row["stl"],
                role=PatchRole(row["role"]),
                box_faces=row["box_faces"],
                refinement=refinement,
            )
        )
    bbox_min = tuple(state.geo_bbox[0])
    bbox_max = tuple(state.geo_bbox[1])
    location = _parse_vec(state.geo_location) or tuple(
        0.5 * (bbox_min[i] + bbox_max[i]) for i in range(3)
    )
    return GeometrySpec(
        bbox_min=(bbox_min[0], bbox_min[1], bbox_min[2]),
        bbox_max=(bbox_max[0], bbox_max[1], bbox_max[2]),
        location_in_mesh=(location[0], location[1], location[2]),
        length_scale=state.geo_length_scale,
        patches=patches,
    )


def _schema_key(entry: FormEntry) -> str:
    """A JS-identifier-safe state var name holding this entry's static schema."""
    return "schema_" + entry.key.replace(":", "_")


def build_app(server: Any = None, *, solver_name: str = "incompressibleFluid") -> Any:
    """Construct the trame Server, state, layout and Save controller. Returns it."""
    from trame.app import get_server  # type: ignore  # untyped optional 'ui' dep
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout  # type: ignore  # untyped dep
    from trame.widgets import client, html, vuetify3 as v3  # type: ignore  # untyped deps
    from trame_client.widgets.core import AbstractElement  # type: ignore  # untyped dep

    class JsonForms(AbstractElement):  # type: ignore[misc]  # untyped base
        """The bundled ``<json-forms>`` client component (JSONForms + Vuetify)."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__("json-forms", **kwargs)
            self._attr_names += ["schema", "uischema", "data"]
            self._event_names += ["change"]

    solver = resolve_solver(solver_name)
    entries: list[FormEntry] = build_forms(solver)
    steps = build_steps(solver, entries)
    choices = build_model_choices(solver)
    required = [c for c in choices if c.required]
    optional = [c for c in choices if not c.required]
    by_key = {e.key: e for e in entries}

    server = get_server() if server is None else server
    state, ctrl = server.state, server.controller

    state.current_step = steps[0].id
    state.ai_panel = True  # right AI drawer open by default (foldable)
    state.target_dir = ""
    state.save_report = None
    state.scaffolded = []
    state.validation_ok = None
    state.findings = []
    # Geometry & mesh stage (STL → blockMesh/snappy dicts).
    state.role_names = _ROLE_NAMES
    state.stl_dir = _DEFAULT_STL_DIR
    state.geometry_patches = []
    state.geometry_status = ""
    state.geo_bbox = None
    state.geo_location = ""
    state.geo_length_scale = 0.0
    state.geo_cell_size = 0.0
    state.mesh_written = []
    for c in optional:
        state[f"sel_{c.name}"] = False
    for entry in entries:
        state[_schema_key(entry)] = entry.schema
        state[entry.state_key] = dict(entry.defaults)

    # ------------------------------------------------------------------ #
    # Controller                                                         #
    # ------------------------------------------------------------------ #

    def _validate_and_store() -> None:
        report = tools.validate_case(solver, state.target_dir)
        state.validation_ok = report.ok
        state.findings = [asdict(r) for r in findings_to_rows(report)]

    def save_case() -> None:
        selected = {c.name for c in optional if state[f"sel_{c.name}"]}
        form_state = {e.key: dict(state[e.state_key]) for e in entries}
        state.current_step = "review"
        try:
            # state_to_case_spec merges/validates field halves and may itself raise.
            spec = cs.state_to_case_spec(entries, form_state, selected)
            result = tools.save_case(solver, spec, state.target_dir)
        except Exception as exc:  # noqa: BLE001 - a save failure must not crash the UI
            # Incomplete/invalid configs — surface each field instead of crashing.
            state.save_report = {"error": str(exc)}
            state.scaffolded = []
            state.validation_ok = False
            state.findings = [asdict(r) for r in review.save_error_rows(exc)]
            return
        state.save_report = result.model_dump()
        # Make the saved case runnable, then validate it.
        state.scaffolded = [str(p) for p in scaffold_runnable_case(state.target_dir)]
        _validate_and_store()

    def revalidate() -> None:
        _validate_and_store()

    def load_geometry() -> None:
        """Read the STLs in ``state.stl_dir`` (an STL folder or case dir) into state."""
        try:
            spec = discover_geometry(state.stl_dir)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
            state.geometry_patches = []
            state.geo_bbox = None
            state.geometry_status = f"Could not read geometry: {exc}"
            return
        # Pre-fill the case target from the STL folder if not already set.
        if not state.target_dir:
            state.target_dir = _case_dir_for(state.stl_dir)
        patch_rows = [_patch_row(p) for p in spec.patches]
        state.geometry_patches = patch_rows
        state.geo_bbox = [list(spec.bbox_min), list(spec.bbox_max)]
        state.geo_location = " ".join(f"{x:g}" for x in spec.location_in_mesh)
        state.geo_length_scale = spec.length_scale
        state.geo_cell_size = round(0.5 * spec.length_scale, 6)
        state.mesh_written = []
        # Seed each boundary-conditions form with the discovered patches so the BC
        # step shows the real patch names (with a role-based BC) instead of a blank
        # property map, and pin them as concrete schema properties so each patch
        # renders as its own titled section. Existing user/AI entries are preserved.
        for entry in entries:
            if entry.kind != "field_bc":
                continue
            seeded = seed_boundary_field(entry, patch_rows, state[entry.state_key])
            state[entry.state_key] = seeded
            names = list(seeded.get("boundaryField", {}).keys())
            state[_schema_key(entry)] = patch_bc_schema(entry, names)
        n = len(spec.patches)
        state.geometry_status = f"Found {n} patch(es) in {state.stl_dir}."

    def write_mesh() -> None:
        """Author blockMeshDict / snappyHexMeshDict / preprocess.yaml from the state."""
        if not state.geometry_patches:
            state.geometry_status = "Scan a case first — no patches loaded."
            return
        try:
            spec = _spec_from_state(state)
            settings = MeshSettings(cell_size=state.geo_cell_size or None)
            written = write_mesh_configs(state.target_dir, spec, settings)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
            state.mesh_written = []
            state.geometry_status = f"Failed to write mesh dicts: {exc}"
            return
        state.mesh_written = [str(p) for p in written]
        state.geometry_status = f"Wrote {len(written)} mesh file(s)."

    ctrl.save_case = save_case
    ctrl.revalidate = revalidate
    ctrl.load_geometry = load_geometry
    ctrl.write_mesh = write_mesh
    ctrl.get_entries = lambda: entries
    ctrl.get_steps = lambda: steps

    build_agent_panel(server, entries, solver)
    sweep_panel = SweepPanel(server, entries, solver, solver_name)

    # ------------------------------------------------------------------ #
    # Layout                                                             #
    # ------------------------------------------------------------------ #

    def _step_header(step: Step) -> None:
        html.Div(step.label, classes="text-h5 font-weight-bold nf-step-title")
        html.Div(
            _STEP_CAPTIONS.get(step.id, ""),
            classes="text-body-2 text-medium-emphasis mb-5",
        )

    def _form_panel(entry: FormEntry) -> None:
        # Owned by an optional model → hide unless selected; else always visible
        # (omit v-show entirely — a bare `v-show` with no expression won't compile).
        panel_kwargs = {}
        if entry.owner_model is not None:
            panel_kwargs["v_show"] = f"sel_{entry.owner_model}"
        with v3.VExpansionPanel(elevation=0, **panel_kwargs):
            with v3.VExpansionPanelTitle():
                html.Span(entry.title)
                if entry.owner_model is not None:
                    v3.VChip(
                        entry.owner_model,
                        size="x-small",
                        color="secondary",
                        variant="tonal",
                        classes="ml-2",
                    )
            with v3.VExpansionPanelText():
                JsonForms(
                    schema=(_schema_key(entry),),
                    data=(entry.state_key,),
                    change=f"{entry.state_key} = $event.data",
                )

    def _model_selector() -> None:
        """Required models (locked on) + optional model toggles."""
        with v3.VCard(variant="outlined", classes="mb-6"):
            with v3.VCardText():
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
                html.Div(
                    "Optional models",
                    classes="text-overline text-medium-emphasis",
                )
                for c in optional:
                    v3.VSwitch(v_model=(f"sel_{c.name}",), label=c.label, inset=True)

    def _geometry_panel() -> None:
        """STL patches → blockMesh / snappy / preprocess dicts."""
        with v3.VCard(variant="outlined", classes="mb-4"):
            with v3.VCardText():
                with v3.VRow(align="center", dense=True):
                    with v3.VCol():
                        v3.VTextField(
                            v_model=("stl_dir",),
                            label="STL folder (triSurface or case dir)",
                            hide_details=True,
                            prepend_inner_icon="mdi-folder-outline",
                        )
                    with v3.VCol(cols="auto"):
                        v3.VBtn(
                            "Scan",
                            click=ctrl.load_geometry,
                            color="primary",
                            variant="tonal",
                            prepend_icon="mdi-magnify",
                        )
                html.Div(
                    "Point at a folder of STLs (or a case dir with"
                    " constant/triSurface/*.stl) and Scan. Adjust patch roles and"
                    " refinement, then Write mesh to author blockMeshDict,"
                    " snappyHexMeshDict and preprocess.yaml.",
                    classes="text-body-2 text-medium-emphasis mt-3",
                )
        # Per-patch role + refinement editor (rows come from the scan).
        with v3.VCard(
            variant="outlined", classes="mb-4", v_show="geometry_patches.length"
        ):
            with v3.VTable(density="compact", hover=True):
                with html.Thead():
                    with html.Tr():
                        html.Th("Patch")
                        html.Th("STL")
                        html.Th("Faces")
                        html.Th("Role")
                        html.Th("Refinement")
                with html.Tbody():
                    with html.Tr(v_for="(p, i) in geometry_patches", key="i"):
                        with html.Td():
                            html.Span("{{ p.name }}", classes="font-weight-medium")
                        with html.Td():
                            html.Span("{{ p.stl }}", classes="text-medium-emphasis")
                        with html.Td():
                            html.Span("{{ p.faces }}", classes="text-medium-emphasis")
                        with html.Td():
                            v3.VSelect(
                                v_model=("p.role",),
                                items=("role_names",),
                                hide_details=True,
                                style="min-width: 130px",
                            )
                        with html.Td():
                            v3.VTextField(
                                v_model=("p.refinement_str",),
                                v_show="p.is_snappy",
                                placeholder="min max",
                                hide_details=True,
                                style="max-width: 110px",
                            )
            # Background-mesh knobs.
            with v3.VCardText():
                with v3.VRow(dense=True):
                    with v3.VCol(cols="6"):
                        v3.VTextField(
                            v_model=("geo_location",),
                            label="locationInMesh (x y z)",
                            hide_details=True,
                        )
                    with v3.VCol(cols="6"):
                        v3.VTextField(
                            v_model=("geo_cell_size", 0.0),
                            label="Background cell size (m, 0 = auto)",
                            type="number",
                            hide_details=True,
                        )
                v3.VBtn(
                    "Write mesh",
                    click=ctrl.write_mesh,
                    color="primary",
                    prepend_icon="mdi-cube-outline",
                    classes="mt-4",
                )
        v3.VAlert(
            text=("geometry_status",),
            type="info",
            variant="tonal",
            v_show="geometry_status",
            classes="mb-2",
        )
        v3.VAlert(
            text=("'Wrote: ' + mesh_written.join(', ')",),
            type="success",
            variant="tonal",
            v_show="mesh_written.length",
        )

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
            "Click 'Save case' to write the case, scaffold Allrun/Allclean and"
            " validate it.",
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
            v_model=("ai_panel", True),
            location="right",
            width=400,
        ):
            with html.Div(classes="d-flex flex-column", style="height: 100%;"):
                v3.VToolbar(title="AI assistant", density="compact", flat=True)
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
                    v3.VProgressLinear(
                        indeterminate=True, v_show="ai_busy", color="secondary"
                    )
                # Composer pinned to the bottom.
                with html.Div(classes="pa-3"):
                    v3.VTextField(
                        v_model=("chat_input",),
                        placeholder="Message the assistant…",
                        hide_details=True,
                        keydown_enter=(ctrl.send_message, "[]"),
                    )
                    v3.VBtn(
                        "Send",
                        click=(ctrl.send_message, "[]"),
                        loading=("ai_busy",),
                        disabled=("!chat_input",),
                        color="secondary",
                        prepend_icon="mdi-send",
                        block=True,
                        classes="mt-2",
                    )

    with SinglePageWithDrawerLayout(server, vuetify_config=_VUETIFY_CONFIG) as layout:
        layout.title.set_text("NeoFOAM case wizard")
        client.Style(_CSS)

        with layout.drawer:
            with v3.VList(nav=True, density="comfortable", color="primary"):
                v3.VListSubheader(solver_name)
                for step in steps:
                    v3.VListItem(
                        title=step.label,
                        prepend_icon=_STEP_ICONS.get(step.id, "mdi-circle-outline"),
                        rounded="lg",
                        active=(f"current_step === '{step.id}'",),
                        click=f"current_step = '{step.id}'",
                    )

        with layout.toolbar:
            v3.VSpacer()
            v3.VTextField(
                v_model=("target_dir",),
                label="Target directory",
                hide_details=True,
                prepend_inner_icon="mdi-folder-arrow-down-outline",
                style="max-width: 340px",
            )
            v3.VBtn(
                "Save case",
                click=ctrl.save_case,
                color="primary",
                variant="flat",
                prepend_icon="mdi-content-save-outline",
                classes="mx-3",
            )
            # Fold / unfold the AI assistant drawer.
            v3.VBtn(
                icon="mdi-robot-happy-outline",
                click="ai_panel = !ai_panel",
                variant="text",
            )

        with layout.root:
            _ai_drawer()

        with layout.content, v3.VContainer(fluid=True, classes="pa-6"):
            # One uniform loop: every step renders its header, its bespoke panel
            # (models / geometry / review) and its schema-generated form panels.
            for step in steps:
                with html.Div(v_show=f"current_step === '{step.id}'"):
                    _step_header(step)
                    if step.id == "models":
                        _model_selector()
                    elif step.id == "geometry":
                        _geometry_panel()
                    elif step.id == "sweep":
                        sweep_panel.render(JsonForms)
                    elif step.id == "review":
                        _review_panel()
                    if step.entry_keys:
                        with v3.VExpansionPanels():
                            for key in step.entry_keys:
                                _form_panel(by_key[key])

    server.enable_module(jsonforms_module)
    return server
