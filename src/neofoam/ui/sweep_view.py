# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The trame layout of the Parameters step: toolbar, palette, canvas, dialogs and tabs.

Pure layout over a :class:`~neofoam.ui.sweep_panel.SweepPanel`, which owns the state
and the controllers; trame is imported where it is used (an optional dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from neofoam.ui.steps import selection_key

if TYPE_CHECKING:
    from neofoam.ui.sweep_panel import SweepPanel

_CSS = """
.nf-sweep-node { font-size: 11px; }
.nf-sweep-node .nf-sweep-header {
  padding: 5px 12px; font-weight: 600; font-size: 12px;
  border-radius: 8px 8px 0 0; color: white; position: relative; text-align: left;
}
.nf-sweep-header-dim { background: #00838F; }
.nf-sweep-header-rule { background: #1F5FBF; }
.nf-sweep-node .port-row { position: relative; padding: 4px 14px; text-align: left; }
.nf-sweep-node .port-row.out { text-align: right; }
.nf-sweep-node .vue-flow__handle.nf-handle-file { border-radius: 2px; }
/* Bounded to the canvas height: unbounded, the (much taller) palette stretches the
   canvas card to its own height and the graph shrinks to an island at the top. */
.nf-sweep-palette { max-height: 62vh; width: 300px; flex: 0 0 300px; overflow-y: auto; }
.nf-sweep-canvas { flex: 1 1 auto; min-width: 0; }
/* Phase L: dimension nodes are a compact overview — the form lives in the
   Configure tab below the canvas, not inside the node. */
.nf-sweep-dim-meta {
  padding: 8px 12px; display: flex; align-items: center; gap: 8px;
  cursor: pointer;
}
.nf-sweep-configure { padding: 12px 16px; }
/* Phone/tablet: the palette stacks above a fixed-height canvas. */
@media (max-width: 959.98px) {
  .nf-sweep-stage { flex-direction: column; }
  .nf-sweep-palette { max-height: 32vh; width: auto; flex: 0 0 auto; }
  .nf-sweep-out-dir { flex: 0 0 100%; }
}
"""


#: Both ends of a valid connection carry the same payload once the ``in:`` /
#: ``out:`` prefixes are stripped and wildcards are normalized — the JS mirror
#: of ``neofoam.tooling.workflow.sweep._normalize_pattern``.
_IS_VALID_CONNECTION = (
    "(c) => {"
    " if (!c.sourceHandle || !c.targetHandle || c.source === c.target)"
    "   return false;"
    " const norm = (s) =>"
    "   s.replace(/^(in|out):/, '').replace(/\\{[^}]+\\}/g, '{*}');"
    " return norm(c.sourceHandle) === norm(c.targetHandle);"
    " }"
)


def render(panel: SweepPanel, json_forms: type) -> tuple[Any, Any]:
    """Build the step panel; returns the canvas editor and the Snakemake graph view."""
    from trame.widgets import client, html  # type: ignore  # untyped deps  # noqa: PLC0415
    from trame.widgets import vuetify3 as v3  # noqa: PLC0415

    client.Style(_CSS)
    _toolbar(panel, html, v3)
    _alerts(v3)
    editor = _stage(panel, html, v3)
    _node_menu(panel, v3)
    _dim_picker_dialog(panel, html, v3)
    _generator_dialog(panel, html, v3)
    _confirm_dialog(panel, v3)
    _load_confirm_dialog(panel, v3)
    return editor, _tabs(panel, json_forms, html, v3)


def _toolbar(panel: SweepPanel, html: Any, v3: Any) -> None:
    """Export target, the export / load / rewire buttons and the case count."""
    # Toolbar: export target + status.
    with v3.VCard(variant="outlined", classes="mb-4"):
        with v3.VCardText():
            with v3.VRow(align="center", dense=True):
                with v3.VCol(classes="nf-sweep-out-dir"):
                    v3.VTextField(
                        v_model=("sweep_out_dir",),
                        label="Sweep directory (default: <target>-sweep)",
                        hide_details=True,
                        prepend_inner_icon="mdi-folder-arrow-down-outline",
                    )
                with v3.VCol(cols="auto"):
                    # V1: Export disabled while any variant is invalid;
                    # V3: a dot badge marks unexported canvas changes.
                    with v3.VBadge(
                        dot=True,
                        color="warning",
                        model_value=("sweep_dirty",),
                    ):
                        v3.VBtn(
                            "Export sweep",
                            click=panel.export,
                            color="primary",
                            prepend_icon="mdi-export-variant",
                            # No target → the default out dir would be a
                            # relative "-sweep" in the launch directory.
                            disabled=("!sweep_valid || !target_dir.trim()",),
                        )
                with v3.VCol(cols="auto"):
                    # V4: reopen a previously exported sweep from the dir.
                    v3.VBtn(
                        "Load",
                        click=panel.load_exported,
                        variant="text",
                        prepend_icon="mdi-folder-open-outline",
                    )
                with v3.VCol(cols="auto"):
                    v3.VBtn(
                        "Rewire",
                        click=panel.rewire,
                        variant="text",
                        prepend_icon="mdi-vector-polyline",
                    )
                v3.VSpacer()
                with v3.VCol(cols="auto"):
                    # V2: factorized case count; warning-colored past the cap.
                    v3.VChip(
                        "{{ sweep_count_label }}",
                        color=("sweep_count_warn ? 'warning' : 'secondary'",),
                        variant="tonal",
                    )
            html.Div(
                "Add configs from the palette as sweep dimensions (the canvas"
                " shows the pipeline), then configure each dimension's variants"
                " in the Configure tab below. Pick the pipeline rules and Export:"
                " the cross product of all variants becomes one Snakemake case"
                " each, cloned from the saved base case.",
                classes="text-body-2 text-medium-emphasis mt-3",
            )


def _alerts(v3: Any) -> None:
    """The save-first hint and the sweep's error / status lines."""
    v3.VAlert(
        "Save the case first — the sweep clones the saved case.",
        type="info",
        variant="tonal",
        classes="mb-3",
        v_show="!scaffolded.length",
    )
    v3.VAlert(
        text=("sweep_error",),
        type="error",
        variant="tonal",
        classes="mb-3",
        v_show="sweep_error",
    )
    v3.VAlert(
        text=("sweep_status",),
        type="success",
        variant="tonal",
        classes="mb-3",
        v_show="sweep_status && !sweep_error",
    )


def _stage(panel: SweepPanel, html: Any, v3: Any) -> Any:
    """Palette + canvas side by side; returns the canvas editor."""
    from trame_flow.widgets.flow import (  # type: ignore  # noqa: PLC0415
        Background,
        Controls,
        CustomNode,
        Handle,
        MiniMap,
        NodeEditor,
    )

    # Palette + canvas.
    with html.Div(classes="d-flex mb-4 nf-sweep-stage", style="gap: 12px; align-items: stretch;"):
        with v3.VCard(variant="outlined", classes="nf-sweep-palette"):
            _palette(panel, v3)
        with v3.VCard(variant="outlined", classes="nf-sweep-canvas"):
            with NodeEditor(
                style="height: 62vh; width: 100%;",
                connection_mode="strict",
                is_valid_connection=(_IS_VALID_CONNECTION,),
                # Right-click on a node opens the delete/disable menu (the
                # raw JS suppresses the browser's own context menu).
                node_context_menu=(
                    "$event.event.preventDefault();"
                    " trigger('nf_sweep_node_menu', [$event.node.id,"
                    " $event.event.clientX, $event.event.clientY])"
                ),
                # Clicking a dimension node configures it in the tab below.
                node_click=(
                    "$event.node && $event.node.type === 'dim' &&"
                    " trigger('nf_sweep_select_cfg', [$event.node.id])"
                ),
            ) as editor:
                Background(gap=12, size=1, pattern_color="#81818a")
                Controls()
                MiniMap()
                _dim_node_template(html, v3, Handle, CustomNode)
                _rule_node_template(html, Handle, CustomNode)
    # The full rule pipeline is wider than VueFlow's default minZoom of 0.5
    # allows fitting; without this, fit_view clips the canvas edges (the
    # first nodes end up under the nav drawer). min_zoom is not an exposed
    # widget attribute — bind the VueFlow prop directly (snakeflow's DAG
    # view uses the same escape hatch).
    editor._attributes["min_zoom"] = ':min-zoom="0.15"'
    return editor


def _node_menu(panel: SweepPanel, v3: Any) -> None:
    """The node context menu, anchored at the right-click position."""
    with v3.VMenu(
        v_model=("sweep_menu_show",),
        target=("[sweep_menu_x, sweep_menu_y]",),
    ):
        with v3.VList(density="compact"):
            v3.VListItem(
                title=("sweep_menu_title",),
                prepend_icon="mdi-delete-outline",
                disabled=("sweep_menu_locked",),
                click=panel.menu_action,
            )


def _tabs(panel: SweepPanel, json_forms: type, html: Any, v3: Any) -> Any:
    """Configure / Parameters / Snakemake graph tabs; returns the graph view."""
    from trame_flow.widgets.flow import Background, Controls, NodeEditor  # noqa: PLC0415

    # Parameters table + Snakemake DAG, as tabs of one card. The tab bodies
    # use v_show (not a window/lazy container) so the DAG editor stays
    # mounted and keeps its graph while the table tab is active.
    with v3.VCard(variant="outlined"):
        with v3.VTabs(
            v_model=("sweep_tab",),
            density="compact",
            color="primary",
        ):
            v3.VTab("Configure", value="configure", prepend_icon="mdi-tune-variant")
            v3.VTab("Parameters", value="table", prepend_icon="mdi-table")
            v3.VTab("Snakemake graph", value="dag", prepend_icon="mdi-graph-outline")
        v3.VDivider()
        with html.Div(v_show="sweep_tab === 'configure'"):
            _configure_tab(panel, json_forms, html, v3)
        with html.Div(v_show="sweep_tab === 'table'"):
            # All combinations, one row per case (no pagination).
            v3.VDataTable(
                headers=("sweep_headers",),
                items=("sweep_rows",),
                density="compact",
                items_per_page=-1,
                hide_default_footer=True,
                fixed_header=True,
                style="max-height: 44vh; overflow-y: auto;",
                v_show="sweep_rows.length",
            )
            v3.VCardText(
                "Add a dimension and name variants to populate the case table.",
                classes="text-medium-emphasis",
                v_show="!sweep_rows.length",
            )
        with html.Div(v_show="sweep_tab === 'dag'"):
            with v3.VCardTitle(classes="text-subtitle-1 d-flex align-center"):
                v3.VSpacer()
                with v3.VBtnToggle(
                    v_model=("sweep_dag_mode",),
                    density="compact",
                    mandatory=True,
                    classes="mx-2",
                    # Switching mode regenerates the graph, so it is a second
                    # snakemake run in disguise.
                    disabled=("sweep_dag_busy",),
                ):
                    v3.VBtn("DAG", value="dag", size="small")
                    v3.VBtn("Rule graph", value="rulegraph", size="small")
                v3.VBtn(
                    "Generate",
                    click=panel.refresh_dag,
                    color="primary",
                    variant="tonal",
                    size="small",
                    prepend_icon="mdi-graph-outline",
                    loading=("sweep_dag_busy",),
                    disabled=("sweep_dag_busy",),
                )
            v3.VAlert(
                text=("sweep_dag_error",),
                type="warning",
                variant="tonal",
                classes="ma-2",
                v_show="sweep_dag_error",
            )
            # V3: the graph reflects the last export, not the live canvas.
            v3.VAlert(
                "Canvas changed since the last export — regenerate to update the graph.",
                type="info",
                variant="tonal",
                density="compact",
                classes="ma-2",
                v_show="sweep_dirty && sweep_exported.length",
            )
            with NodeEditor(style="height: 44vh; width: 100%;") as dag_view:
                Background(gap=12, size=1, pattern_color="#81818a")
                Controls()
    # A wide sweep DAG needs a zoom below VueFlow's default min of 0.5 for
    # fitView to frame it.
    dag_view._attributes["min_zoom"] = ':min-zoom="0.05"'
    return dag_view


def _configure_tab(panel: SweepPanel, json_forms: type, html: Any, v3: Any) -> None:
    """The Configure tab: pick a dimension, edit its variants' form here.

    This is where the config forms live (Phase L) — full-width below the
    canvas instead of cramped inside a node. The dimension selector chips
    mirror the canvas; the form binds to the active dimension's selected
    variant and edits flow back through the ``nf_sweep_edit`` trigger.
    """
    with html.Div(classes="nf-sweep-configure"):
        v3.VCardText(
            "Add a dimension from the palette, then configure its variants here.",
            classes="text-medium-emphasis",
            v_show="!sweep_cfg_chips.length",
        )
        with html.Div(v_show="sweep_cfg_chips.length"):
            # Dimension selector — one chip per dimension on the canvas.
            with html.Div(
                classes="d-flex flex-wrap align-center mb-3",
                style="gap: 6px;",
            ):
                html.Span("Configuring", classes="text-caption text-medium-emphasis mr-1")
                v3.VChip(
                    "{{ c.title }} · {{ c.count }}",
                    v_for="c in sweep_cfg_chips",
                    key=("c.value",),
                    size="small",
                    color="teal",
                    variant=("c.value === sweep_cfg_dim ? 'flat' : 'outlined'",),
                    click=(panel.select_cfg, "[c.value]"),
                )
            # Variant management (moved off the node).
            with html.Div(
                classes="d-flex align-center flex-wrap mb-2",
                style="gap: 6px; max-width: 720px;",
            ):
                v3.VSelect(
                    model_value=("sweep_cfg_selected",),
                    items=("sweep_cfg_variants",),
                    label="variant",
                    density="compact",
                    hide_details=True,
                    style="max-width: 220px;",
                    update_modelValue=(panel.cfg_variant_select, "[$event]"),
                )
                v3.VBtn(
                    icon="mdi-plus",
                    size="small",
                    variant="text",
                    click=panel.cfg_variant_add,
                )
                v3.VBtn(
                    icon="mdi-delete",
                    size="small",
                    variant="text",
                    click=panel.cfg_variant_delete,
                )
                v3.VBtn(
                    "Generate series",
                    size="small",
                    variant="tonal",
                    prepend_icon="mdi-format-list-numbered",
                    click=panel.cfg_open_generator,
                )
            with html.Div(
                classes="d-flex align-center mb-4",
                style="gap: 6px; max-width: 720px;",
            ):
                v3.VTextField(
                    model_value=("sweep_cfg_rename",),
                    label="rename variant",
                    density="compact",
                    hide_details=True,
                    style="max-width: 220px;",
                    update_modelValue=(panel.cfg_rename_buffer, "[$event]"),
                )
                v3.VBtn(
                    icon="mdi-check",
                    size="small",
                    variant="text",
                    click=panel.cfg_variant_rename,
                )
            # V1: the selected variant's validation error, if any.
            v3.VAlert(
                text=("sweep_cfg_error",),
                type="error",
                variant="tonal",
                density="compact",
                classes="mb-3",
                v_show="sweep_cfg_error",
            )
            # The config's form, full width, bound to the selected variant.
            # The change handler mirrors the edit client-side (no echo) and
            # notifies Python via the shared trigger.
            with html.Div(classes="nodrag nowheel", v_if="sweep_cfg_dim"):
                json_forms(
                    schema=("sweep_cfg_schema",),
                    data=("sweep_cfg_data",),
                    change="sweep_cfg_data = $event.data;"
                    " trigger('nf_sweep_edit', ['dim:' + sweep_cfg_dim,"
                    " $event.data])",
                )


def _confirm_dialog(panel: SweepPanel, v3: Any) -> None:
    """V2: confirm exporting a large factorized sweep (over the cap)."""
    with v3.VDialog(v_model=("sweep_confirm_show",), max_width=460):
        with v3.VCard():
            v3.VCardTitle("Large sweep", classes="text-subtitle-1")
            v3.VCardText("{{ sweep_count_label }} — that is a lot of runs. Export anyway?")
            with v3.VCardActions():
                v3.VSpacer()
                v3.VBtn("Cancel", click="sweep_confirm_show = false")
                v3.VBtn(
                    "Export anyway",
                    click=panel.confirm_export,
                    color="warning",
                    variant="tonal",
                )


def _load_confirm_dialog(panel: SweepPanel, v3: Any) -> None:
    """V4: confirm that a load replaces the forms with the sweep's base case."""
    with v3.VDialog(v_model=("sweep_load_confirm_show",), max_width=460):
        with v3.VCard():
            v3.VCardTitle("Reopen sweep", classes="text-subtitle-1")
            v3.VCardText(
                "Loading replaces the wizard forms with this sweep's base case"
                " ({{ sweep_load_base }}) — unsaved edits are lost. Continue?"
            )
            with v3.VCardActions():
                v3.VSpacer()
                v3.VBtn("Cancel", click="sweep_load_confirm_show = false")
                v3.VBtn(
                    "Load anyway",
                    click=panel.confirm_load,
                    color="warning",
                    variant="tonal",
                )


def _dim_picker_dialog(panel: SweepPanel, html: Any, v3: Any) -> None:
    """The "add dimension: pick the parameters to vary" dialog."""
    with v3.VDialog(v_model=("sweep_pick_show",), max_width=560):
        with v3.VCard():
            v3.VCardTitle("Add dimension: {{ sweep_pick_title }}", classes="text-subtitle-1")
            with v3.VCardText():
                v3.VSelect(
                    v_model=("sweep_pick_selected",),
                    items=("sweep_pick_options",),
                    item_title="title",
                    item_value="value",
                    label="Parameters to vary",
                    multiple=True,
                    chips=True,
                    closable_chips=True,
                    hide_details=True,
                )
                html.Div(
                    "The node shows only the picked parameters; everything"
                    " else stays inherited from the base case. Leave empty"
                    " to edit the whole form.",
                    classes="text-body-2 text-medium-emphasis mt-3",
                )
            with v3.VCardActions():
                v3.VSpacer()
                v3.VBtn("Cancel", click="sweep_pick_show = false")
                v3.VBtn(
                    "Add",
                    click=panel.confirm_dim,
                    color="primary",
                    variant="tonal",
                )


def _generator_dialog(panel: SweepPanel, html: Any, v3: Any) -> None:
    """The variant-series generator dialog (list / linear / log spacing)."""
    with v3.VDialog(v_model=("sweep_gen_show",), max_width=560):
        with v3.VCard():
            v3.VCardTitle(
                "Generate variants: {{ sweep_gen_title }}",
                classes="text-subtitle-1",
            )
            with v3.VCardText():
                v3.VSelect(
                    v_model=("sweep_gen_param",),
                    items=("sweep_gen_params",),
                    item_title="title",
                    item_value="value",
                    label="Parameter",
                    hide_details=True,
                    classes="mb-3",
                )
                with v3.VBtnToggle(
                    v_model=("sweep_gen_mode",),
                    density="compact",
                    mandatory=True,
                    classes="mb-3",
                ):
                    v3.VBtn("Values", value="list", size="small")
                    v3.VBtn("Linear", value="linear", size="small")
                    v3.VBtn("Log", value="log", size="small")
                v3.VTextField(
                    v_model=("sweep_gen_values",),
                    label="Values (comma or space separated, e.g. 1e-5 2e-5 4e-5)",
                    hide_details=True,
                    v_show="sweep_gen_mode === 'list'",
                )
                with v3.VRow(dense=True, v_show="sweep_gen_mode !== 'list'"):
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("sweep_gen_min",),
                            label="min",
                            hide_details=True,
                        )
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("sweep_gen_max",),
                            label="max",
                            hide_details=True,
                        )
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("sweep_gen_count",),
                            label="count",
                            hide_details=True,
                        )
                v3.VCheckboxBtn(
                    v_model=("sweep_gen_replace",),
                    label="Replace existing variants",
                    density="compact",
                )
                v3.VAlert(
                    text=("sweep_gen_error",),
                    type="error",
                    variant="tonal",
                    density="compact",
                    v_show="sweep_gen_error",
                )
            with v3.VCardActions():
                v3.VSpacer()
                v3.VBtn("Cancel", click="sweep_gen_show = false")
                v3.VBtn(
                    "Generate",
                    click=panel.generate_variants,
                    color="primary",
                    variant="tonal",
                )


class _PaletteSection(NamedTuple):
    """One palette group: the state arrays behind its rows and what a click does."""

    items: str
    on_canvas: str
    click: Any
    subtitle: Any


def _palette_section(panel: SweepPanel, v3: Any, section: _PaletteSection) -> None:
    """One palette group: add/remove toggle rows over an on-canvas array."""
    items, on_canvas, click, subtitle = section
    # A row owned by a gated model follows its `sel_<model>` switch, like the form
    # panels; the lookup names each state var so Vue tracks the selection.
    entries = (*panel._dims.values(), *panel._field_dims.values())
    owners = sorted({e.owner_model for e in entries if e.owner_model is not None})
    selected = ", ".join(f"'{owner}': {selection_key(owner)}" for owner in owners)
    with v3.VListItem(
        v_for=f"item in {items}",
        v_show=f"!item.owner || ({{{selected}}})[item.owner]",
        key="item.value",
        click=(click, "[item.value]"),
        title=("item.title",),
        subtitle=subtitle,
    ):
        with v3.Template(v_slot_append=True):
            v3.VIcon(
                icon=(
                    f"{on_canvas}.includes(item.value)"
                    " ? 'mdi-minus-circle-outline'"
                    " : 'mdi-plus-circle-outline'",
                ),
                color=(f"{on_canvas}.includes(item.value) ? 'error' : 'primary'",),
                size="small",
            )


def _palette(panel: SweepPanel, v3: Any) -> None:
    """The config + rule palette (left of the canvas)."""
    with v3.VList(density="compact", nav=True):
        v3.VListSubheader("Config dimensions")
        _palette_section(
            panel,
            v3,
            _PaletteSection(
                items="sweep_config_palette",
                on_canvas="sweep_dims_on_canvas",
                click=panel._on_palette_config,
                subtitle=("item.owner ? 'model: ' + item.owner : ''",),
            ),
        )
        v3.VDivider(classes="my-2")
        v3.VListSubheader("Fields / boundary conditions")
        _palette_section(
            panel,
            v3,
            _PaletteSection(
                items="sweep_field_palette",
                on_canvas="sweep_field_on_canvas",
                click=panel._on_palette_field,
                subtitle=("item.owner ? 'model: ' + item.owner : 'per-case'",),
            ),
        )
        v3.VDivider(classes="my-2")
        v3.VListSubheader("Mesh dimension")
        _palette_section(
            panel,
            v3,
            _PaletteSection(
                items="sweep_mesh_palette",
                on_canvas="sweep_mesh_on_canvas",
                click=panel._on_palette_mesh,
                subtitle="re-meshes per variant",
            ),
        )
        v3.VDivider(classes="my-2")
        v3.VListSubheader("Pipeline rules")
        with v3.VListItem(
            v_for="r in sweep_rule_palette",
            key="r.name",
            title=("r.title",),
            subtitle=("r.required ? 'required' : ''",),
        ):
            with v3.Template(v_slot_prepend=True):
                v3.VCheckboxBtn(
                    model_value=("r.enabled",),
                    disabled=("r.required",),
                    density="compact",
                    update_modelValue=(panel.toggle_rule, "[r.name, $event]"),
                )


def _dim_node_template(html: Any, v3: Any, handle: Any, custom_node: Any) -> None:
    """A dimension node: a compact overview chip (Phase L).

    The config's form no longer lives in the node — it is edited in the
    Configure tab below the canvas. The node shows only the config title,
    its variant count and the ``cfg:`` source handle; clicking it selects
    the dimension for the Configure tab.
    """
    with custom_node("dim"), html.Div(classes="nf-sweep-node"):
        with html.Div("{{ props.data.label }}", classes="nf-sweep-header nf-sweep-header-dim"):
            handle(type="source", position="right", id=("'cfg:' + props.data.dim",))
        with html.Div(classes="nf-sweep-dim-meta"):
            v3.VChip(
                "{{ Object.keys(props.data.entries).length }} variant(s)",
                size="x-small",
                color="teal",
                variant="tonal",
            )
            v3.VSpacer()
            # V1: a red badge when the selected/any variant is invalid.
            with html.Span(v_show="props.data.error"):
                v3.VIcon(
                    "mdi-alert-circle",
                    size="x-small",
                    color="error",
                )
                v3.VTooltip(
                    text=("props.data.error",),
                    activator="parent",
                    location="top",
                )
            v3.VIcon(
                "mdi-pencil-outline",
                size="x-small",
                color="grey",
                v_show="!props.data.error",
            )


def _rule_node_template(html: Any, handle: Any, custom_node: Any) -> None:
    """A fixed pipeline node: config ports per dimension + typed file ports."""
    with custom_node("rule"), html.Div(classes="nf-sweep-node"):
        html.Div("{{ props.data.label }}", classes="nf-sweep-header nf-sweep-header-rule")
        with html.Div(v_for="dim in props.data.cfg_dims", key=("dim",), classes="port-row"):
            handle(type="target", position="left", id=("'cfg:' + dim",))
            html.Span("{{ dim }}")
        with html.Div(v_for="f in props.data.inputs", key=("f",), classes="port-row"):
            handle(
                type="target",
                position="left",
                id=("'in:' + f",),
                classes="nf-handle-file",
            )
            html.Span("{{ f.split('/').pop() }}")
        with html.Div(v_for="f in props.data.outputs", key=("f",), classes="port-row out"):
            html.Span("{{ f.split('/').pop() }}")
            handle(
                type="source",
                position="right",
                id=("'out:' + f",),
                classes="nf-handle-file",
            )
