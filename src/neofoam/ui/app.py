# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Trame case wizard: sidebar step nav + JSONForms panels + model-aware save.

Left drawer lists the wizard steps (``current_step``); the main area mounts every
step once, shown via ``v_show`` so navigation never unmounts a form and wipes its
state. The Setup step toggles optional models (``sel_<model>``); a config panel owned
by an unselected optional model is hidden and skipped on save. Save aggregates the
live form state through :func:`neofoam.ui.case_spec.state_to_case_spec` and writes via
:func:`neofoam.mcp.tools.save_case`.
"""

from __future__ import annotations

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
from neofoam.ui.steps import build_model_choices, build_steps

_ROLE_NAMES = [r.value for r in PatchRole]

# Default STL folder — a manual-testing convenience so the geometry step is
# pre-populated and one click away from a scan.
_DEFAULT_STL_DIR = (
    "/home/henning/libsAndApps/NeoFOAM/.claude/worktrees/feat+e2eWorkflow/"
    "playground/hx/constant/triSurface"
)


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
    """Parse a ``"min max"`` refinement string; ``None`` when unparseable."""
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
    from trame.widgets import html, vuetify3 as v3  # type: ignore  # untyped 'ui' deps
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

    state.current_step = "setup"
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

    def _form_panel(entry: FormEntry) -> None:
        # Owned by an optional model → hide unless selected; else always visible
        # (omit v-show entirely — a bare `v-show` with no expression won't compile).
        panel_kwargs = {}
        if entry.owner_model is not None:
            panel_kwargs["v_show"] = f"sel_{entry.owner_model}"
        with v3.VExpansionPanel(**panel_kwargs):
            v3.VExpansionPanelTitle(entry.title)
            with v3.VExpansionPanelText():
                JsonForms(
                    schema=(_schema_key(entry),),
                    data=(entry.state_key,),
                    change=f"{entry.state_key} = $event.data",
                )

    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("NeoFOAM case wizard")

        with layout.drawer:
            with v3.VList(nav=True, density="compact"):
                for step in steps:
                    v3.VListItem(
                        title=step.label,
                        active=(f"current_step === '{step.id}'",),
                        click=f"current_step = '{step.id}'",
                    )

        with layout.toolbar:
            v3.VSpacer()
            v3.VTextField(
                v_model=("target_dir",),
                label="Target directory",
                density="compact",
                hide_details=True,
                style="max-width: 320px",
            )
            v3.VBtn("Save case", click=ctrl.save_case, color="primary", classes="ml-3")
            # Fold / unfold the AI assistant drawer.
            v3.VBtn(
                icon="mdi-robot-happy-outline",
                click="ai_panel = !ai_panel",
                variant="text",
                classes="ml-2",
            )

        # Right-hand foldable AI chat drawer (multi-turn, fills the forms).
        with layout.root:
            with v3.VNavigationDrawer(
                v_model=("ai_panel", True),
                location="right",
                width=400,
            ):
                with html.Div(
                    classes="d-flex flex-column",
                    style="height: 100%;",
                ):
                    v3.VToolbar(
                        title="AI assistant",
                        density="compact",
                        flat=True,
                    )
                    # Scrolling transcript.
                    with html.Div(
                        classes="flex-grow-1 pa-3",
                        style="overflow-y: auto;",
                    ):
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
                            color=(
                                "m.role === 'user' ? 'primary' : 'surface-variant'",
                            ),
                        ):
                            html.Div(
                                "{{ m.content }}",
                                style="white-space: pre-wrap; font-size: 0.9rem;",
                            )
                        v3.VProgressLinear(
                            indeterminate=True,
                            v_show="ai_busy",
                            color="secondary",
                        )
                    # Composer pinned to the bottom.
                    with html.Div(classes="pa-3"):
                        v3.VTextField(
                            v_model=("chat_input",),
                            placeholder="Message the assistant…",
                            hide_details=True,
                            variant="outlined",
                            density="compact",
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

        with layout.content, v3.VContainer(fluid=True):
            # Setup — model selection.
            with html.Div(v_show="current_step === 'setup'"):
                v3.VCardTitle("Choose models")
                v3.VListSubheader("Required (always on)")
                with v3.VList(density="compact"):
                    for c in required:
                        v3.VListItem(title=c.label, prepend_icon="mdi-lock")
                v3.VListSubheader("Optional")
                for c in optional:
                    v3.VSwitch(
                        v_model=(f"sel_{c.name}",),
                        label=c.label,
                        density="compact",
                        hide_details=True,
                        color="primary",
                    )

            # Geometry & mesh — STL patches → blockMesh / snappy / preprocess dicts.
            with html.Div(v_show="current_step === 'geometry'"):
                v3.VCardTitle("Geometry & mesh")
                # STL folder picker (a triSurface folder or a case dir) + Scan.
                with v3.VRow(align="center", classes="mb-2"):
                    with v3.VCol():
                        v3.VTextField(
                            v_model=("stl_dir",),
                            label="STL folder (triSurface or case dir)",
                            density="compact",
                            hide_details=True,
                            prepend_inner_icon="mdi-folder-outline",
                        )
                    with v3.VCol(cols="auto"):
                        v3.VBtn(
                            "Scan STL folder",
                            click=ctrl.load_geometry,
                            variant="tonal",
                            prepend_icon="mdi-magnify",
                        )
                v3.VAlert(
                    "Point at a folder of STLs (or a case dir with"
                    " constant/triSurface/*.stl), then Scan. Adjust patch roles /"
                    " refinement and click Write mesh to author blockMeshDict,"
                    " snappyHexMeshDict and preprocess.yaml.",
                    type="info",
                    variant="tonal",
                    density="compact",
                    classes="mb-3",
                )
                # Per-patch role + refinement editor (rows come from the scan).
                with v3.VTable(density="compact", v_show="geometry_patches.length"):
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
                                html.Span("{{ p.name }}")
                            with html.Td():
                                html.Span("{{ p.stl }}", classes="text-medium-emphasis")
                            with html.Td():
                                html.Span(
                                    "{{ p.faces }}", classes="text-medium-emphasis"
                                )
                            with html.Td():
                                v3.VSelect(
                                    v_model=("p.role",),
                                    items=("role_names",),
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    style="min-width: 130px",
                                )
                            with html.Td():
                                v3.VTextField(
                                    v_model=("p.refinement_str",),
                                    v_show="p.is_snappy",
                                    placeholder="min max",
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    style="max-width: 110px",
                                )
                # Background-mesh knobs.
                with v3.VRow(classes="mt-3", v_show="geometry_patches.length"):
                    with v3.VCol(cols="6"):
                        v3.VTextField(
                            v_model=("geo_location",),
                            label="locationInMesh (x y z)",
                            density="compact",
                            hide_details=True,
                        )
                    with v3.VCol(cols="6"):
                        v3.VTextField(
                            v_model=("geo_cell_size", 0.0),
                            label="Background cell size (m, 0 = auto)",
                            type="number",
                            density="compact",
                            hide_details=True,
                        )
                v3.VBtn(
                    "Write mesh",
                    click=ctrl.write_mesh,
                    color="primary",
                    prepend_icon="mdi-cube-outline",
                    classes="mt-3",
                    v_show="geometry_patches.length",
                )
                v3.VAlert(
                    text=("geometry_status",),
                    type="info",
                    variant="outlined",
                    density="compact",
                    classes="mt-3",
                    v_show="geometry_status",
                )
                v3.VAlert(
                    text=("'Wrote: ' + mesh_written.join(', ')",),
                    type="success",
                    variant="outlined",
                    density="compact",
                    classes="mt-2",
                    v_show="mesh_written.length",
                )

            # Form steps.
            for step in steps:
                if step.id in ("setup", "geometry", "review"):
                    continue
                with html.Div(v_show=f"current_step === '{step.id}'"):
                    with v3.VExpansionPanels(multiple=True):
                        for key in step.entry_keys:
                            _form_panel(by_key[key])

            # Review — validate_case findings + scaffolded runnable case.
            with html.Div(v_show="current_step === 'review'"):
                with v3.VRow(align="center", classes="mb-2"):
                    v3.VCardTitle("Review & run")
                    v3.VSpacer()
                    v3.VBtn(
                        "Re-validate",
                        click=ctrl.revalidate,
                        variant="tonal",
                        prepend_icon="mdi-refresh",
                    )
                # Prompt to save first.
                v3.VAlert(
                    "Click 'Save case' to write the case, scaffold Allrun/Allclean and validate.",
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
                    density="compact",
                    classes="mt-3",
                    v_show="scaffolded.length",
                )

    server.enable_module(jsonforms_module)
    return server
