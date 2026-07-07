# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The Parameters step: a trame-flow canvas that sweeps configs over variants.

The step is laid out as a workbench:

- A **palette** sidebar left of the canvas lists the sweepable configs and the
  rule library's rules — clicking a config opens a *parameter picker* (which
  fields does this dimension vary?) and adds it as a dimension node; clicking
  again removes it. The rule checkboxes select which pipeline rules the sweep
  is composed of (required rules are locked).
- The **canvas** holds the dimension nodes (each a live view into one
  ``params.yaml`` section, rendering the picked parameters — or the config's
  whole form — for the selected variant; payloads stay full, unrendered keys
  are inherited from the base case) and the pipeline nodes derived from the
  packaged rule library (:mod:`neofoam.workflow.rules`). A node's *series
  generator* creates variants from a value list or a linear/log range,
  auto-named like ``nu1e-05``.
- Below the canvas, a tabbed card holds the **parameters table** — every
  combination, one row per case, with the varied parameters' concrete values —
  and the **Snakemake graph** of the exported workflow (``snakemake --dag`` /
  ``--rulegraph``, rendered via :mod:`neofoam.workflow.snakemake_dag`).
- Right-clicking a canvas node opens a context menu: remove the dimension /
  disable the rule (required rules are locked).

Only this module talks to trame; the canvas data model, the export and the
DAG rendering live UI-free in :mod:`neofoam.workflow`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from neofoam.io.pydantic_schema import slice_schema
from neofoam.ui.forms import FormEntry, build_field_forms, build_mesh_forms
from neofoam.ui.sweep_model import DimensionState, SweepModel, series_values
from neofoam.workflow.rules import DEFAULT_ENABLED, default_registry
from neofoam.workflow.snakemake_dag import dag_graph
from neofoam.workflow.sweep import (
    SweepDimension,
    autowire,
    dim_node,
    export_sweep,
    load_sweep,
    rule_nodes,
)
from neofoam.workflow.sweep_runner import config_classes_by_name

#: Above this many cases the count chip turns warning-colored and Export asks
#: for confirmation (a factorized sweep grows fast — a guard against a typo in
#: a series generator spawning thousands of runs).
_CASE_WARN_THRESHOLD = 64

__all__ = ["SweepPanel"]

#: Rules every pipeline needs (mirrors ``RuleRegistry.plan``) — locked in the
#: palette so they cannot be toggled off.
_REQUIRED_RULES = frozenset({"all", "setup", "solve", "setup_mesh"})


def _dim_name(node_id: str) -> str:
    """The model dimension name behind a ``dim:<name>`` canvas node id."""
    return node_id[len("dim:") :] if node_id.startswith("dim:") else node_id


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
.nf-sweep-palette { width: 300px; flex: 0 0 300px; overflow-y: auto; }
.nf-sweep-canvas { flex: 1 1 auto; min-width: 0; }
/* Phase L: dimension nodes are a compact overview — the form lives in the
   Configure tab below the canvas, not inside the node. */
.nf-sweep-dim-meta {
  padding: 8px 12px; display: flex; align-items: center; gap: 8px;
  cursor: pointer;
}
.nf-sweep-configure { padding: 12px 16px; }
"""


#: Both ends of a valid connection carry the same payload once the ``in:`` /
#: ``out:`` prefixes are stripped and wildcards are normalized — the JS mirror
#: of ``neofoam.workflow.sweep._normalize_pattern``.
_IS_VALID_CONNECTION = (
    "(c) => {"
    " if (!c.sourceHandle || !c.targetHandle || c.source === c.target)"
    "   return false;"
    " const norm = (s) =>"
    "   s.replace(/^(in|out):/, '').replace(/\\{[^}]+\\}/g, '{*}');"
    " return norm(c.sourceHandle) === norm(c.targetHandle);"
    " }"
)


class SweepPanel:
    """State, controllers and canvas of the parameter-sweep step."""

    def __init__(
        self, server: Any, entries: list[FormEntry], solver: Any, solver_name: str
    ):
        self._server = server
        self._solver = solver
        self._solver_name = solver_name
        # Sweepable dimensions: dict-kind configs only (field halves are merged
        # at save time and their schemas mutate with the geometry scan).
        self._dims: dict[str, FormEntry] = {
            e.config_name: e for e in entries if e.kind == "dict"
        }
        # Sweepable mesh dicts (blockMesh/snappy) — sources of the keyed ``mesh``
        # dimension, kept out of the physics wizard (see forms._MESH_FILES).
        self._mesh_dims: dict[str, FormEntry] = {
            e.config_name: e for e in build_mesh_forms(solver)
        }
        # Whole ``0/<field>`` configs — sweepable boundary/field values (e.g. the
        # inlet velocity). Regular per-case dimensions (setup.json), unlike the
        # keyed mesh dicts; the wizard forms only carry their split halves.
        self._field_dims: dict[str, FormEntry] = {
            e.config_name: e for e in build_field_forms(solver)
        }
        self._editor: Any = None
        self._dag_view: Any = None
        self._fit_task: Any = None
        self._registry = default_registry()
        self._dag_generated = False
        # Config classes for live variant validation (V1), computed once.
        self._classes = config_classes_by_name(solver)
        # The sweep definition (dimensions/variants/enabled rules/dirty) — pure,
        # trame-free; the panel renders it into state + canvas nodes (M1).
        self._model = SweepModel()

        state = server.state
        state.sweep_dim_pick = None
        state.sweep_config_palette = [
            {"title": e.title, "value": name, "owner": e.owner_model or ""}
            for name, e in sorted(self._dims.items())
        ]
        state.sweep_dim_choices = [  # kept: headless drivers pick by value
            {"title": e.title, "value": name} for name, e in sorted(self._dims.items())
        ]
        state.sweep_dims_on_canvas = []
        # Mesh-source palette: the blockMesh/snappy dicts feeding the keyed mesh
        # dimension, and which of them are currently on the canvas.
        state.sweep_mesh_palette = [
            {"title": e.title, "value": name}
            for name, e in sorted(self._mesh_dims.items())
        ]
        state.sweep_mesh_on_canvas = []
        # Field / boundary-condition palette (whole 0/<field> configs).
        state.sweep_field_palette = [
            {"title": e.title, "value": name, "owner": e.owner_model or ""}
            for name, e in sorted(self._field_dims.items())
        ]
        state.sweep_field_on_canvas = []
        state.sweep_rule_palette = self._rule_palette()
        state.sweep_out_dir = ""
        state.sweep_status = ""
        state.sweep_error = ""
        state.sweep_exported = []
        state.sweep_case_count = 0
        # V1 validation: overall validity gate + the active variant's error.
        state.sweep_valid = True
        state.sweep_cfg_error = ""
        # V2 case-count factorization: e.g. "2 × 3 = 6 case(s)" + warn flag.
        state.sweep_count_label = "0 case(s)"
        state.sweep_count_warn = False
        state.sweep_confirm_show = False
        # V3 staleness: the canvas differs from the last export.
        state.sweep_dirty = False
        state.sweep_headers = []
        state.sweep_rows = []
        state.sweep_dag_mode = "dag"
        state.sweep_dag_error = ""
        state.sweep_tab = "configure"
        # Configure tab (Phase L): the forms live here, not inside the nodes.
        # ``sweep_cfg_dim`` is the dimension being configured; the rest mirror
        # its live node data (schema + variants + the selected variant's data).
        state.sweep_cfg_dim = ""
        state.sweep_cfg_title = ""
        state.sweep_cfg_chips = []
        state.sweep_cfg_schema = {}
        state.sweep_cfg_variants = []
        state.sweep_cfg_selected = ""
        state.sweep_cfg_rename = ""
        state.sweep_cfg_data = {}
        state.sweep_cfg_fields = []
        # Right-click node context menu (position + single action item).
        state.sweep_menu_show = False
        state.sweep_menu_x = 0
        state.sweep_menu_y = 0
        state.sweep_menu_node = ""
        state.sweep_menu_title = ""
        state.sweep_menu_locked = False
        # "Add dimension" parameter picker (palette click).
        state.sweep_pick_show = False
        state.sweep_pick_config = ""
        state.sweep_pick_title = ""
        state.sweep_pick_options = []
        state.sweep_pick_selected = []
        # Variant series generator (in-node button).
        state.sweep_gen_show = False
        state.sweep_gen_node = ""
        state.sweep_gen_title = ""
        state.sweep_gen_params = []
        state.sweep_gen_param = None
        state.sweep_gen_mode = "list"
        state.sweep_gen_values = ""
        state.sweep_gen_min = ""
        state.sweep_gen_max = ""
        state.sweep_gen_count = "3"
        state.sweep_gen_replace = False
        state.sweep_gen_error = ""

        ctrl = server.controller
        ctrl.sweep_add_dimension = self.add_dimension
        ctrl.sweep_add_cad_dimension = self.add_cad_dimension
        ctrl.sweep_add_mesh_source = self.add_mesh_source
        ctrl.sweep_remove_mesh_source = self.remove_mesh_source
        ctrl.sweep_add_field_source = self.add_field_source
        ctrl.sweep_remove_dimension = self.remove_dimension
        ctrl.sweep_toggle_dimension = self.toggle_dimension
        ctrl.sweep_toggle_rule = self.toggle_rule
        ctrl.sweep_export = self.export
        ctrl.sweep_confirm_export = self.confirm_export
        ctrl.sweep_load = self.load_exported
        ctrl.sweep_refresh_dag = self.refresh_dag
        ctrl.sweep_rewire = self.rewire
        ctrl.sweep_get_nodes = lambda: list(self._editor.nodes) if self._editor else []
        ctrl.sweep_get_dag_nodes = lambda: (
            list(self._dag_view.nodes) if self._dag_view else []
        )
        # The variant ops are node-template event handlers; registering them on
        # the controller too gives headless drivers (tests, scripted examples)
        # the same seam the browser uses.
        ctrl.sweep_variant_select = self._on_variant_select
        ctrl.sweep_variant_add = self._on_variant_add
        ctrl.sweep_variant_delete = self._on_variant_delete
        ctrl.sweep_variant_rename = self._on_variant_rename
        ctrl.sweep_rename_buffer = self._on_rename_buffer
        ctrl.sweep_variant_edit = self._on_variant_edit
        ctrl.sweep_node_menu = self._on_node_menu
        ctrl.sweep_menu_action = self.menu_action
        ctrl.sweep_open_dim_picker = self.open_dim_picker
        ctrl.sweep_confirm_dim = self.confirm_dim
        ctrl.sweep_open_generator = self.open_generator
        ctrl.sweep_generate_variants = self.generate_variants
        # Configure-tab seams (Phase L): pick a dimension to configure and
        # drive its variants from the tab instead of the node.
        ctrl.sweep_select_cfg = self.select_cfg
        ctrl.sweep_cfg_variant_select = self.cfg_variant_select
        ctrl.sweep_cfg_variant_add = self.cfg_variant_add
        ctrl.sweep_cfg_variant_delete = self.cfg_variant_delete
        ctrl.sweep_cfg_variant_rename = self.cfg_variant_rename
        ctrl.sweep_cfg_rename_buffer = self.cfg_rename_buffer
        ctrl.sweep_cfg_open_generator = self.cfg_open_generator
        # The in-node JSONForms change handler dispatches through a trigger
        # (it needs a raw JS expression to also mirror the edit client-side).
        server.trigger("nf_sweep_edit")(self._on_variant_edit)
        # The canvas right-click dispatches through a trigger too (the JS side
        # must also suppress the browser's own context menu).
        server.trigger("nf_sweep_node_menu")(self._on_node_menu)
        # Clicking a dimension node selects it for the Configure tab.
        server.trigger("nf_sweep_select_cfg")(self.select_cfg_from_node)
        # Regenerate the DAG when the graph mode flips (only once one exists).
        server.state.change("sweep_dag_mode")(self._on_dag_mode)
        # Re-frame the DAG when its tab becomes visible (hidden = unmeasured).
        server.state.change("sweep_tab")(self._on_tab_change)

    # -- state helpers -------------------------------------------------------

    def _notify(self, message: str = "", error: str = "") -> None:
        self._server.state.sweep_status = message
        self._server.state.sweep_error = error

    def _rule_palette(self) -> list[dict[str, Any]]:
        """The rule-library palette rows (registry order, required rules locked)."""
        return [
            {
                "name": name,
                "title": self._registry.get(name).title or name,
                "required": name in _REQUIRED_RULES,
                "enabled": name in self._model.enabled,
            }
            for name in self._registry.names
        ]

    def _dim_nodes(self) -> list[dict[str, Any]]:
        return [n for n in self._editor.nodes if n.get("type") == "dim"]

    # -- model → canvas rendering (M1) -----------------------------------------

    def _dim_node_dict(self, state: DimensionState, index: int) -> dict[str, Any]:
        """A compact canvas node built from a model dimension (Phase L shape)."""
        seed = next(iter(state.entries.values())) if state.entries else {}
        spec = SweepDimension(
            name=state.name, title=state.title, schema=state.schema, seed=dict(seed)
        )
        node = dim_node(
            f"dim:{state.name}",
            spec,
            entries={k: dict(v) for k, v in state.entries.items()},
            selected=state.selected,
            x=0.0,
            y=40.0 + 120.0 * index,
        )
        node["data"]["fields"] = list(state.fields)
        node["data"]["rename"] = state.rename or state.selected
        node["data"]["error"] = ""
        node["style"]["width"] = "220px"
        return node

    def _rebuild_graph(self) -> None:
        """Rebuild the whole canvas (dimension + pipeline nodes) from the model."""
        states = self._model.sorted_dims()
        dim_nodes = [self._dim_node_dict(s, i) for i, s in enumerate(states)]
        rules = rule_nodes([s.name for s in states], enabled=self._model.enabled)
        nodes = [*dim_nodes, *rules]
        self._editor.graph = {"nodes": nodes, "edges": autowire(nodes)}
        self._server.state.sweep_dims_on_canvas = sorted(s.name for s in states)
        self._server.state.sweep_mesh_on_canvas = self._model.mesh_sources()
        self._server.state.sweep_field_on_canvas = sorted(
            s.name for s in states if s.name in self._field_dims
        )

    def _write_dim_node(self, name: str, *, echo: bool) -> None:
        """Sync one dimension node's server-side data from the model.

        ``echo=False`` mutates the node dict in place without pushing to the
        client (a variant *edit* changes no visible node content — the form
        lives in the Configure tab — so the snakeflow no-echo rule still holds).
        """
        node = self._editor.get_node(f"dim:{name}") if self._editor else None
        if node is None:
            return
        state = self._model.dims[name]
        node["data"]["entries"] = {k: dict(v) for k, v in state.entries.items()}
        node["data"]["selected"] = state.selected
        node["data"]["rename"] = state.rename or state.selected
        if echo:
            self._editor.update_node(f"dim:{name}", data=node["data"])

    def _refresh_overview(self) -> None:
        """Push the model's derivations (count, table, validation, dirty) to state.

        The table lists every combination (one row per case) with a ``validation``
        cell (V1); the toolbar shows the factorized case count (V2) and a
        staleness flag (V3); each dimension node gets its error badge (V1).
        """
        state = self._server.state
        ov = self._model.overview(self._classes, _CASE_WARN_THRESHOLD)
        state.sweep_case_count = ov.case_count
        state.sweep_count_label = ov.count_label
        state.sweep_count_warn = ov.warn
        state.sweep_valid = ov.valid
        state.sweep_headers = ov.headers
        state.sweep_rows = ov.rows
        state.sweep_cfg_error = ov.errors.get(state.sweep_cfg_dim, {}).get(
            state.sweep_cfg_selected, ""
        )
        self._badge_dim_nodes(ov.errors)
        state.sweep_dirty = self._model.dirty

    def _badge_dim_nodes(self, errors: dict[str, dict[str, str]]) -> None:
        """Set each dimension node's ``data.error`` (V1) — the selected
        variant's message if it fails, else the first failing variant's."""
        for node in self._dim_nodes():
            dim = str(node["data"]["dim"])
            dim_errors = errors.get(dim, {})
            if not dim_errors:
                message = ""
            else:
                selected = node["data"].get("selected")
                message = dim_errors.get(selected) or next(iter(dim_errors.values()))
            if node["data"].get("error", "") != message:
                node["data"]["error"] = message
                self._editor.update_node(node["id"], data=node["data"])

    def _sync_setup_ports(self) -> None:
        """Re-derive the canvas from the model (setup ports, wiring, overview)."""
        self._rebuild_graph()
        self._refresh_overview()
        self._sync_cfg_mirror()

    # -- configure tab (Phase L) ------------------------------------------------

    def _cfg_node_id(self) -> str:
        """The dimension node id currently bound to the Configure tab."""
        return f"dim:{self._server.state.sweep_cfg_dim}"

    def _cfg_chips(self) -> list[dict[str, Any]]:
        """Selector chips for the Configure tab — one per dimension (by title)."""
        return [
            {"title": d.title, "value": d.name, "count": len(d.entries)}
            for d in sorted(self._model.sorted_dims(), key=lambda d: d.title)
        ]

    def _sync_cfg_mirror(self) -> None:
        """Mirror the active dimension's model state into the Configure state.

        Every call re-pushes the selected variant's form payload. The no-echo
        rule (an in-form edit must not fight the user's cursor) is realized by
        ``_on_variant_edit`` deliberately *not* calling this method — the client
        already mirrors that edit locally.
        """
        state = self._server.state
        state.sweep_cfg_chips = self._cfg_chips()
        dim = self._model.dims.get(state.sweep_cfg_dim)
        if dim is None:
            # The active dimension was removed (or none picked yet) — fall back
            # to the first dimension on the canvas, else clear the tab.
            states = self._model.sorted_dims()
            if states:
                state.sweep_cfg_dim = states[0].name
                dim = states[0]
            else:
                state.sweep_cfg_dim = ""
                state.sweep_cfg_title = ""
                state.sweep_cfg_schema = {}
                state.sweep_cfg_variants = []
                state.sweep_cfg_selected = ""
                state.sweep_cfg_rename = ""
                state.sweep_cfg_data = {}
                state.sweep_cfg_fields = []
                return
        state.sweep_cfg_title = dim.title
        state.sweep_cfg_schema = dim.schema
        state.sweep_cfg_variants = list(dim.entries)
        state.sweep_cfg_selected = dim.selected
        state.sweep_cfg_rename = dim.rename or dim.selected
        state.sweep_cfg_fields = list(dim.fields)
        # V1: the active variant's validation message (switching variants must
        # refresh the badge even without a full table recompute).
        state.sweep_cfg_error = self._model.variant_error(dim.name, self._classes)
        state.sweep_cfg_data = dict(dim.entries[dim.selected])

    def select_cfg(self, name: str) -> None:
        """Configure-tab selector: make ``name`` the active dimension."""
        self._server.state.sweep_cfg_dim = name
        self._sync_cfg_mirror()

    def select_cfg_from_node(self, node_id: str, *_: Any) -> None:
        """Canvas click on a dimension node → configure it in the tab."""
        node = self._editor.get_node(node_id) if self._editor else None
        if node is None or node.get("type") != "dim":
            return
        state = self._server.state
        state.sweep_cfg_dim = str(node["data"]["dim"])
        state.sweep_tab = "configure"
        self._sync_cfg_mirror()

    def cfg_variant_select(self, name: str) -> None:
        # The node handler already mirrors the active dimension (the Configure
        # tab always operates on it), so no extra sync is needed here.
        self._on_variant_select(self._cfg_node_id(), name)

    def cfg_variant_add(self) -> None:
        self._on_variant_add(self._cfg_node_id())

    def cfg_variant_delete(self) -> None:
        self._on_variant_delete(self._cfg_node_id())

    def cfg_variant_rename(self) -> None:
        self._on_variant_rename(self._cfg_node_id())

    def cfg_rename_buffer(self, value: str) -> None:
        self._on_rename_buffer(self._cfg_node_id(), value)
        self._server.state.sweep_cfg_rename = value

    def cfg_open_generator(self) -> None:
        self.open_generator(self._cfg_node_id())

    def _fit_soon(self, editor: Any) -> None:
        """Fit the view once the client has measured the new nodes."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no event loop (headless/startup) — nothing to frame

        async def _fit() -> None:
            # VueFlow frames nodes from their *measured* sizes — wait until the
            # client has laid the freshly pushed graph out.
            await asyncio.sleep(0.6)
            with self._server.state:
                editor.fit_view()

        self._fit_task = loop.create_task(_fit())

    # -- canvas actions --------------------------------------------------------

    def add_dimension(
        self, name: str | None = None, fields: list[str] | None = None
    ) -> None:
        """Add a config as a dimension node, seeded from the live form.

        With ``fields``, the node renders only those top-level parameters (a
        display slice of the schema); the variant payloads stay *full* — the
        unrendered keys are inherited from the seed and survive form edits —
        because the export path validates whole config payloads.
        """
        state = self._server.state
        entry = self._dims.get(name or state.sweep_dim_pick or "")
        if entry is None:
            self._notify(error="Pick a config dimension to add.")
            return
        if entry.owner_model is not None and not state[f"sel_{entry.owner_model}"]:
            self._notify(
                error=f"'{entry.title}' belongs to the unselected model"
                f" '{entry.owner_model}' — enable it in the Models step first."
            )
            return
        if self._model.has(entry.config_name):
            self._notify(error=f"'{entry.title}' is already on the canvas.")
            return
        schema = entry.schema
        if fields:
            known = set(schema.get("properties", {}))
            unknown = [f for f in fields if f not in known]
            if unknown:
                self._notify(
                    error=f"unknown parameter(s) {unknown} for '{entry.title}'"
                )
                return
            schema = slice_schema(schema, fields)
        seed = dict(state[entry.state_key]) or dict(entry.defaults)
        self._model.add_dimension(
            entry.config_name,
            title=entry.title,
            schema=schema,
            seed=seed,
            fields=list(fields or []),
        )
        self._rebuild_graph()
        self._refresh_overview()
        # Configure the freshly added dimension straight away.
        state.sweep_cfg_dim = entry.config_name
        self._sync_cfg_mirror()
        self._editor.fit_view()  # bring the new node into the viewport
        self._notify(message=f"Added dimension '{entry.title}'.")

    def add_cad_dimension(self, model_path: str, params: dict[str, float]) -> None:
        """Add the CAD geometry axis (reserved dim ``cad``) to the canvas.

        Called by the FoamCAD step's "Add to sweep": ``params`` are the selected
        parametric aliases and their current values. The CAD dimension is an
        independent, STL-producing axis (it composes with the mesh axis as
        ``cad × mesh``) and is configured as a numeric parameter map.
        """
        state = self._server.state
        if self._model.has("cad"):
            self._notify(error="A CAD dimension is already on the canvas.")
            return
        self._model.add_cad_dimension(
            "cad", title="CAD geometry", model_path=model_path, params=params
        )
        self._rebuild_graph()
        self._refresh_overview()
        state.sweep_cfg_dim = "cad"
        self._sync_cfg_mirror()
        self._editor.fit_view()
        self._notify(message="Added CAD dimension.")

    def _config_seed(self, entry: FormEntry) -> dict[str, Any]:
        """The full config from the saved base case (schema-defaults fallback).

        Sweeping applies *whole* configs (``apply_configs`` validates and rewrites
        the file), so a dimension must be seeded with the base case's complete
        payload — the same ``model_dump(by_alias=True, exclude_none=True)`` shape
        the headless examples use — not a partial override. When the base case has
        no such file yet, fall back to the schema defaults so the dimension is
        still addable (the user fills it in).
        """
        cls = self._classes.get(entry.config_name)
        target = self._server.state.target_dir
        if cls is not None and target:
            try:
                loaded = cls.load(case_dir=Path(target))
                seed = loaded.model_dump(by_alias=True, exclude_none=True)
                seed.pop("FoamFile", None)
                return seed
            except Exception:  # noqa: BLE001 - file absent/unreadable: use defaults
                pass
        return dict(entry.defaults)

    def add_mesh_source(self, config_name: str) -> None:
        """Add a mesh dict (blockMesh/snappy) as a source of the keyed mesh dim."""
        state = self._server.state
        entry = self._mesh_dims.get(config_name)
        if entry is None:
            self._notify(error=f"Unknown mesh source '{config_name}'.")
            return
        try:
            self._model.add_mesh_source(
                config_name,
                title=entry.title,
                schema=entry.schema,
                seed=self._config_seed(entry),
            )
        except ValueError as exc:
            self._notify(error=str(exc))
            return
        self._rebuild_graph()
        self._refresh_overview()
        state.sweep_cfg_dim = "mesh"
        self._sync_cfg_mirror()
        self._editor.fit_view()
        self._notify(message=f"Added mesh source '{entry.title}'.")

    def add_field_source(self, config_name: str) -> None:
        """Add a whole ``0/<field>`` config as a (per-case) sweep dimension.

        A field config (e.g. ``u_field_config``) is a regular dimension — it
        applies in the per-case clone via ``setup.json``, not the keyed mesh
        mini-case — so it goes through the normal :meth:`SweepModel.add_dimension`
        path, seeded from the saved base case. Gated on its owner model like any
        optional-model config (``t_field_config`` needs Boussinesq enabled).
        """
        state = self._server.state
        entry = self._field_dims.get(config_name)
        if entry is None:
            self._notify(error=f"Unknown field '{config_name}'.")
            return
        if entry.owner_model is not None and not state[f"sel_{entry.owner_model}"]:
            self._notify(
                error=f"'{entry.title}' belongs to the unselected model"
                f" '{entry.owner_model}' — enable it in the Models step first."
            )
            return
        try:
            self._model.add_dimension(
                config_name,
                title=entry.title,
                schema=entry.schema,
                seed=self._config_seed(entry),
            )
        except ValueError as exc:
            self._notify(error=str(exc))
            return
        self._rebuild_graph()
        self._refresh_overview()
        state.sweep_cfg_dim = config_name
        self._sync_cfg_mirror()
        self._editor.fit_view()
        self._notify(message=f"Added field dimension '{entry.title}'.")

    def _on_palette_field(self, config_name: str) -> None:
        """Palette click on a field/BC config: toggle it as a sweep dimension."""
        if self._model.has(config_name):
            self.remove_dimension(config_name)
        else:
            self.add_field_source(config_name)

    def remove_mesh_source(self, config_name: str) -> None:
        """Remove a mesh source; drops the mesh dimension if it was the last."""
        entry = self._mesh_dims.get(config_name)
        try:
            self._model.remove_mesh_source(config_name)
        except ValueError as exc:
            self._notify(error=str(exc))
            return
        self._rebuild_graph()
        self._refresh_overview()
        self._sync_cfg_mirror()
        self._fit_soon(self._editor)
        title = entry.title if entry else config_name
        self._notify(message=f"Removed mesh source '{title}'.")

    def _on_palette_mesh(self, config_name: str) -> None:
        """Palette click on a mesh source: toggle it on the keyed mesh dimension."""
        if config_name in self._model.mesh_sources():
            self.remove_mesh_source(config_name)
        else:
            self.add_mesh_source(config_name)

    def open_dim_picker(self, name: str) -> None:
        """Open the "which parameters vary?" dialog for a palette config."""
        state = self._server.state
        entry = self._dims.get(name)
        if entry is None:
            return
        if entry.owner_model is not None and not state[f"sel_{entry.owner_model}"]:
            self._notify(
                error=f"'{entry.title}' belongs to the unselected model"
                f" '{entry.owner_model}' — enable it in the Models step first."
            )
            return
        state.sweep_pick_config = entry.config_name
        state.sweep_pick_title = entry.title
        state.sweep_pick_options = [
            {
                "title": prop.get("title", key) if isinstance(prop, dict) else key,
                "value": key,
            }
            for key, prop in entry.schema.get("properties", {}).items()
        ]
        state.sweep_pick_selected = []
        state.sweep_pick_show = True

    def confirm_dim(self) -> None:
        """Add the picker's config with the selected parameters (empty = all)."""
        state = self._server.state
        state.sweep_pick_show = False
        self.add_dimension(
            state.sweep_pick_config, fields=list(state.sweep_pick_selected) or None
        )

    def remove_dimension(self, name: str) -> None:
        """Remove a dimension from the canvas (its variants are dropped)."""
        entry = self._dims.get(name)
        try:
            self._model.remove_dimension(name)
        except ValueError:
            self._notify(error=f"'{name}' is not on the canvas.")
            return
        self._rebuild_graph()
        self._refresh_overview()
        self._sync_cfg_mirror()
        self._fit_soon(self._editor)  # re-frame the canvas around what remains
        title = entry.title if entry else name
        self._notify(message=f"Removed dimension '{title}'.")

    def toggle_dimension(self, name: str) -> None:
        """Headless seam: add the config as a full-form dimension, or remove it."""
        if self._model.has(name):
            self.remove_dimension(name)
        else:
            self.add_dimension(name)

    def _on_palette_config(self, name: str) -> None:
        """Palette click: open the parameter picker, or remove the dimension."""
        if self._model.has(name):
            self.remove_dimension(name)
        else:
            self.open_dim_picker(name)

    def toggle_rule(self, name: str, enabled: bool) -> None:
        """Enable/disable a rule of the library; the pipeline nodes re-derive."""
        current = self._model.enabled
        chosen = [n for n in self._registry.names if n in current and n != name]
        if enabled:
            chosen = [n for n in self._registry.names if n in {*chosen, name}]
        try:
            self._registry.plan(chosen)
        except ValueError as exc:
            self._notify(error=str(exc))
            self._server.state.sweep_rule_palette = self._rule_palette()
            self._server.state.dirty("sweep_rule_palette")  # revert the checkbox
            return
        self._model.set_enabled(chosen)
        self._rebuild_graph()
        self._server.state.sweep_rule_palette = self._rule_palette()
        self._refresh_overview()
        self._fit_soon(self._editor)
        self._notify(message=f"Rule '{name}' {'enabled' if enabled else 'disabled'}.")

    def rewire(self) -> None:
        """Re-derive the setup ports and edges (e.g. after deleting a node)."""
        self._sync_setup_ports()
        self._notify(message="Rewired the canvas.")

    def load_exported(self) -> None:
        """V4: read an exported sweep back onto the canvas (from the dir field).

        Rebuilds the dimension nodes (full variant payloads), restores the
        enabled rules and re-derives the pipeline — so a prior study can be
        reopened, re-validated (V1 badges anything that no longer validates) and
        extended. Dimensions with no matching config on this solver (e.g. a mesh
        dimension while the mesh canvas node is still API-only) are reported and
        skipped.
        """
        state = self._server.state
        out_dir = state.sweep_out_dir or f"{state.target_dir}-sweep"
        try:
            loaded = load_sweep(out_dir)
        except (ValueError, OSError) as exc:
            self._notify(error=f"Could not load sweep: {exc}")
            return

        states: list[DimensionState] = []
        skipped: list[str] = []
        for dim, variants in sorted(loaded.dimensions.items()):
            if dim in ("cad", "mesh"):
                continue  # the cad/mesh axes are restored below (not solver-config)
            entry = self._dims.get(dim)
            if entry is None or not variants:
                skipped.append(dim)
                continue
            states.append(
                DimensionState(
                    name=dim,
                    title=entry.title,
                    schema=entry.schema,  # whole form: picked slices aren't persisted
                    entries={k: dict(v) for k, v in variants.items()},
                    selected=next(iter(variants)),
                )
            )

        # Restore the keyed mesh axis: its variants are config-name-keyed
        # (``{config_name: payload}``), so rebuild the combined schema from the
        # source configs present across all variants (mirror of add_mesh_source).
        mesh_variants = loaded.dimensions.get("mesh")
        if mesh_variants:
            sources = sorted(
                {name for payload in mesh_variants.values() for name in payload}
            )
            mesh_props = {
                name: {
                    **self._mesh_dims[name].schema,
                    "title": self._mesh_dims[name].title,
                }
                for name in sources
                if name in self._mesh_dims
            }
            states.append(
                DimensionState(
                    name="mesh",
                    title="Mesh",
                    schema={"type": "object", "properties": mesh_props},
                    entries={k: dict(v) for k, v in mesh_variants.items()},
                    selected=next(iter(mesh_variants)),
                    kind="mesh",
                )
            )

        # Restore the CAD axis: it has no solver-config entry in `self._dims`, so
        # rebuild its numeric DimensionState directly from the loaded variants +
        # the header's model path (mirror of SweepModel.add_cad_dimension).
        cad_variants = loaded.dimensions.get("cad")
        if cad_variants and loaded.cad_model:
            aliases = sorted({a for payload in cad_variants.values() for a in payload})
            cad_schema: dict[str, Any] = {
                "type": "object",
                "properties": {a: {"type": "number", "title": a} for a in aliases},
            }
            states.append(
                DimensionState(
                    name="cad",
                    title="CAD geometry",
                    schema=cad_schema,
                    entries={k: dict(v) for k, v in cad_variants.items()},
                    selected=next(iter(cad_variants)),
                    fields=list(aliases),
                    kind="cad",
                    model_path=loaded.cad_model,
                )
            )

        # Restore the enabled rules (fall back to the defaults if the header's
        # selection no longer resolves against this registry).
        try:
            self._registry.plan(loaded.enabled)
            enabled = list(loaded.enabled)
        except ValueError:
            enabled = list(DEFAULT_ENABLED)

        self._model.load(states, enabled)  # replaces the canvas; starts clean
        self._rebuild_graph()

        state.sweep_out_dir = str(out_dir)
        state.sweep_rule_palette = self._rule_palette()
        state.sweep_cfg_dim = states[0].name if states else ""
        self._refresh_overview()
        self._sync_cfg_mirror()
        self._fit_soon(self._editor)

        note = f"Loaded {len(states)} dimension(s) from {out_dir}"
        if skipped:
            note += f" — skipped unsupported dimension(s): {', '.join(skipped)}"
        self._notify(message=note, error="")

    def confirm_export(self) -> None:
        """The confirm dialog's "Export anyway" action (V2, over threshold)."""
        self._server.state.sweep_confirm_show = False
        self.export(force=True)

    def export(self, force: bool = False) -> None:
        """Write the runnable sweep workflow next to the saved base case."""
        state = self._server.state
        if not state.scaffolded:
            self._notify(error="Save the case first — the sweep clones the saved case.")
            return
        # V1: never export an invalid sweep (the button is also disabled, but
        # headless callers and the confirm path reach here too).
        if not state.sweep_valid:
            self._notify(
                error="Some variants are invalid — fix them before exporting"
                " (see the Configure tab and the table's validation column)."
            )
            return
        # V2: a large factorized sweep asks for confirmation once.
        if not force and state.sweep_count_warn:
            state.sweep_confirm_show = True
            return
        out_dir = state.sweep_out_dir or f"{state.target_dir}-sweep"
        try:
            result = export_sweep(
                out_dir,
                solver_name=self._solver_name,
                base_case=Path(state.target_dir).resolve(),
                dimensions=self._model.to_dimensions(),
                classes=self._classes,
                cad=self._model.cad_dimensions(),
                enabled=self._model.enabled,
            )
        except ValueError as exc:
            self._notify(error=str(exc))
            return
        state.sweep_out_dir = str(result.out_dir)
        state.sweep_exported = sorted(
            str(p.relative_to(result.out_dir))
            for p in (
                result.sweep_csv,
                result.params_yaml,
                result.snakefile,
                *result.configs,
            )
        )
        # V3: the on-disk sweep now matches the canvas.
        self._model.mark_exported()
        self._refresh_overview()
        self._notify(
            message=f"Exported {state.sweep_case_count} case(s) to {result.out_dir} —"
            f" run them with: cd '{result.out_dir}' && snakemake -j4"
        )

    def refresh_dag(self) -> None:
        """Export the sweep, run ``snakemake --dag|--rulegraph`` and render it."""
        state = self._server.state
        self.export()
        # Nothing was written if export errored or is awaiting confirmation.
        if state.sweep_error or state.sweep_confirm_show:
            return
        try:
            nodes, edges = dag_graph(state.sweep_out_dir, state.sweep_dag_mode)
        except (RuntimeError, ValueError) as exc:
            state.sweep_dag_error = str(exc)
            self._dag_view.clear_graph()
            return
        state.sweep_dag_error = ""
        self._dag_generated = True
        self._dag_view.graph = {"nodes": nodes, "edges": edges}
        self._fit_soon(self._dag_view)

    def _on_dag_mode(self, sweep_dag_mode: str, **_: Any) -> None:
        if self._dag_generated:
            self.refresh_dag()

    def _on_tab_change(self, sweep_tab: str, **_: Any) -> None:
        # The DAG editor is hidden (v-show) while the table tab is active, so
        # VueFlow could not measure its pane — re-frame once it is visible.
        if sweep_tab == "dag" and self._dag_generated:
            self._fit_soon(self._dag_view)
        elif sweep_tab == "configure":
            self._sync_cfg_mirror()

    # -- node context menu -------------------------------------------------------

    def _on_node_menu(self, node_id: str, x: float = 0.0, y: float = 0.0) -> None:
        """Right-click on a canvas node: open the delete/disable menu."""
        node = self._editor.get_node(node_id) if self._editor else None
        if node is None:
            return
        state = self._server.state
        data = node["data"]
        if node.get("type") == "dim":
            state.sweep_menu_title = f"Remove dimension '{data['label']}'"
            state.sweep_menu_locked = False
        else:
            rule = str(data["rule"])
            if rule in _REQUIRED_RULES:
                state.sweep_menu_title = f"'{rule}' is required"
                state.sweep_menu_locked = True
            else:
                state.sweep_menu_title = f"Disable rule '{data['label']}'"
                state.sweep_menu_locked = False
        state.sweep_menu_node = node_id
        state.sweep_menu_x, state.sweep_menu_y = x, y
        state.sweep_menu_show = True

    def menu_action(self) -> None:
        """Run the context menu's action on the node it was opened for."""
        state = self._server.state
        state.sweep_menu_show = False
        node = self._editor.get_node(state.sweep_menu_node) if self._editor else None
        if node is None or state.sweep_menu_locked:
            return
        if node.get("type") == "dim":
            self.remove_dimension(str(node["data"]["dim"]))
        else:
            self.toggle_rule(str(node["data"]["rule"]), False)

    # -- variant series generator -------------------------------------------------

    def open_generator(self, node_id: str) -> None:
        """Open the variant-series dialog for a dimension node."""
        state = self._server.state
        dim = self._model.dims.get(_dim_name(node_id))
        if dim is None:
            return
        params = dim.numeric_params()
        if not params:
            self._notify(
                error=f"'{dim.title}' has no numeric top-level parameter to"
                " generate a series for."
            )
            return
        state.sweep_gen_node = node_id
        state.sweep_gen_title = dim.title
        state.sweep_gen_params = params
        state.sweep_gen_param = params[0]["value"]
        state.sweep_gen_error = ""
        state.sweep_gen_show = True

    def generate_variants(self) -> None:
        """Create one variant per series value on the dialog's dimension."""
        state = self._server.state
        name = _dim_name(state.sweep_gen_node)
        param = state.sweep_gen_param
        if not self._model.has(name) or not param:
            state.sweep_gen_show = False
            return
        try:
            values = series_values(
                state.sweep_gen_mode,
                state.sweep_gen_values,
                state.sweep_gen_min,
                state.sweep_gen_max,
                state.sweep_gen_count,
            )
        except ValueError as exc:
            state.sweep_gen_error = str(exc)
            return
        count = self._model.generate_series(
            name, param, values, replace=state.sweep_gen_replace
        )
        self._write_dim_node(name, echo=True)
        self._refresh_overview()
        self._sync_cfg_mirror()
        state.sweep_gen_error = ""
        state.sweep_gen_show = False
        self._notify(
            message=f"Generated {count} '{param}' variant(s) on"
            f" '{self._model.dims[name].title}'."
        )

    # -- variant handlers -------------------------------------------------------
    #
    # These mutate the model, then sync the affected dimension node. A field
    # *edit* changes no visible node content (the form is in the Configure tab),
    # so it is not echoed to the client (snakeflow no-echo rule); structural
    # changes echo so the variant-count chip refreshes.

    def _on_variant_edit(self, node_id: str, payload: dict[str, Any]) -> None:
        name = _dim_name(node_id)
        if not self._model.has(name):
            return
        self._model.variant_edit(name, payload)
        self._write_dim_node(name, echo=False)
        self._refresh_overview()

    def _on_rename_buffer(self, node_id: str, value: str) -> None:
        name = _dim_name(node_id)
        if self._model.has(name):
            self._model.set_rename_buffer(name, value)

    def _sync_cfg_if_active(self, name: str) -> None:
        """Re-mirror the Configure tab if ``name`` is the dimension it shows.

        A structural node-seam mutation (add/delete/rename/select) must keep the
        ``sweep_cfg_*`` mirror in sync when it hits the active dimension — the
        node seam is otherwise equivalent to the browser's Configure path (and
        could leave the mirror pointing at a deleted/renamed variant).
        """
        if name == self._server.state.sweep_cfg_dim:
            self._sync_cfg_mirror()

    def _on_variant_select(self, node_id: str, name: str) -> None:
        dim = _dim_name(node_id)
        if self._model.has(dim):
            self._model.variant_select(dim, name)
            self._write_dim_node(dim, echo=True)
            self._sync_cfg_if_active(dim)

    def _on_variant_add(self, node_id: str) -> None:
        name = _dim_name(node_id)
        if not self._model.has(name):
            return
        self._model.variant_add(name)
        self._write_dim_node(name, echo=True)
        self._refresh_overview()
        self._sync_cfg_if_active(name)

    def _on_variant_delete(self, node_id: str) -> None:
        name = _dim_name(node_id)
        if not self._model.has(name):
            return
        try:
            self._model.variant_delete(name)
        except ValueError as exc:
            self._notify(error=str(exc))
            return
        self._write_dim_node(name, echo=True)
        self._refresh_overview()
        self._sync_cfg_if_active(name)

    def _on_variant_rename(self, node_id: str) -> None:
        name = _dim_name(node_id)
        if not self._model.has(name):
            return
        try:
            self._model.variant_rename(name, self._model.dims[name].rename)
        except ValueError as exc:
            self._notify(error=str(exc))
            return
        self._write_dim_node(name, echo=True)
        self._refresh_overview()
        self._sync_cfg_if_active(name)

    # -- UI ---------------------------------------------------------------------

    def render(self, json_forms: type) -> None:
        """Build the step panel (called inside the app's layout context).

        Args:
            json_forms: The app's ``JsonForms`` widget class (the bundled
                ``<json-forms>`` component), reused inside the dimension nodes.
        """
        from trame.widgets import client, html, vuetify3 as v3  # type: ignore  # untyped deps
        from trame_flow.widgets.flow import (  # type: ignore  # untyped optional dep
            Background,
            Controls,
            CustomNode,
            Handle,
            MiniMap,
            NodeEditor,
        )

        client.Style(_CSS)

        # Toolbar: export target + status.
        with v3.VCard(variant="outlined", classes="mb-4"):
            with v3.VCardText():
                with v3.VRow(align="center", dense=True):
                    with v3.VCol():
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
                                click=self.export,
                                color="primary",
                                prepend_icon="mdi-export-variant",
                                disabled=("!sweep_valid",),
                            )
                    with v3.VCol(cols="auto"):
                        # V4: reopen a previously exported sweep from the dir.
                        v3.VBtn(
                            "Load",
                            click=self.load_exported,
                            variant="text",
                            prepend_icon="mdi-folder-open-outline",
                        )
                    with v3.VCol(cols="auto"):
                        v3.VBtn(
                            "Rewire",
                            click=self.rewire,
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

        # Palette + canvas.
        with html.Div(classes="d-flex mb-4", style="gap: 12px; align-items: stretch;"):
            with v3.VCard(variant="outlined", classes="nf-sweep-palette"):
                self._palette()
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
                    self._dim_node_template(html, v3, Handle, CustomNode)
                    self._rule_node_template(html, Handle, CustomNode)
        # The node context menu, anchored at the right-click position.
        with v3.VMenu(
            v_model=("sweep_menu_show",),
            target=("[sweep_menu_x, sweep_menu_y]",),
        ):
            with v3.VList(density="compact"):
                v3.VListItem(
                    title=("sweep_menu_title",),
                    prepend_icon="mdi-delete-outline",
                    disabled=("sweep_menu_locked",),
                    click=self.menu_action,
                )
        self._dim_picker_dialog(html, v3)
        self._generator_dialog(html, v3)
        self._confirm_dialog(v3)
        # The full rule pipeline is wider than VueFlow's default minZoom of 0.5
        # allows fitting; without this, fit_view clips the canvas edges (the
        # first nodes end up under the nav drawer). min_zoom is not an exposed
        # widget attribute — bind the VueFlow prop directly (snakeflow's DAG
        # view uses the same escape hatch).
        editor._attributes["min_zoom"] = ':min-zoom="0.15"'
        self._editor = editor
        seed_nodes = rule_nodes([], enabled=self._model.enabled)
        editor.graph = {"nodes": seed_nodes, "edges": autowire(seed_nodes)}

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
                v3.VTab(
                    "Snakemake graph", value="dag", prepend_icon="mdi-graph-outline"
                )
            v3.VDivider()
            with html.Div(v_show="sweep_tab === 'configure'"):
                self._configure_tab(json_forms, html, v3)
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
                    ):
                        v3.VBtn("DAG", value="dag", size="small")
                        v3.VBtn("Rule graph", value="rulegraph", size="small")
                    v3.VBtn(
                        "Generate",
                        click=self.refresh_dag,
                        color="primary",
                        variant="tonal",
                        size="small",
                        prepend_icon="mdi-graph-outline",
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
                    "Canvas changed since the last export — regenerate to update"
                    " the graph.",
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
        self._dag_view = dag_view

    def _configure_tab(self, json_forms: type, html: Any, v3: Any) -> None:
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
                    html.Span(
                        "Configuring", classes="text-caption text-medium-emphasis mr-1"
                    )
                    v3.VChip(
                        "{{ c.title }} · {{ c.count }}",
                        v_for="c in sweep_cfg_chips",
                        key=("c.value",),
                        size="small",
                        color="teal",
                        variant=("c.value === sweep_cfg_dim ? 'flat' : 'outlined'",),
                        click=(self.select_cfg, "[c.value]"),
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
                        update_modelValue=(self.cfg_variant_select, "[$event]"),
                    )
                    v3.VBtn(
                        icon="mdi-plus",
                        size="small",
                        variant="text",
                        click=self.cfg_variant_add,
                    )
                    v3.VBtn(
                        icon="mdi-delete",
                        size="small",
                        variant="text",
                        click=self.cfg_variant_delete,
                    )
                    v3.VBtn(
                        "Generate series",
                        size="small",
                        variant="tonal",
                        prepend_icon="mdi-format-list-numbered",
                        click=self.cfg_open_generator,
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
                        update_modelValue=(self.cfg_rename_buffer, "[$event]"),
                    )
                    v3.VBtn(
                        icon="mdi-check",
                        size="small",
                        variant="text",
                        click=self.cfg_variant_rename,
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

    def _confirm_dialog(self, v3: Any) -> None:
        """V2: confirm exporting a large factorized sweep (over the cap)."""
        with v3.VDialog(v_model=("sweep_confirm_show",), max_width=460):
            with v3.VCard():
                v3.VCardTitle("Large sweep", classes="text-subtitle-1")
                v3.VCardText(
                    "{{ sweep_count_label }} — that is a lot of runs. Export anyway?"
                )
                with v3.VCardActions():
                    v3.VSpacer()
                    v3.VBtn("Cancel", click="sweep_confirm_show = false")
                    v3.VBtn(
                        "Export anyway",
                        click=self.confirm_export,
                        color="warning",
                        variant="tonal",
                    )

    def _dim_picker_dialog(self, html: Any, v3: Any) -> None:
        """The "add dimension: pick the parameters to vary" dialog."""
        with v3.VDialog(v_model=("sweep_pick_show",), max_width=560):
            with v3.VCard():
                v3.VCardTitle(
                    "Add dimension: {{ sweep_pick_title }}", classes="text-subtitle-1"
                )
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
                        click=self.confirm_dim,
                        color="primary",
                        variant="tonal",
                    )

    def _generator_dialog(self, html: Any, v3: Any) -> None:
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
                        click=self.generate_variants,
                        color="primary",
                        variant="tonal",
                    )

    def _palette(self) -> None:
        """The config + rule palette (left of the canvas)."""
        from trame.widgets import vuetify3 as v3  # untyped dep, ignored in render()

        with v3.VList(density="compact", nav=True):
            v3.VListSubheader("Config dimensions")
            with v3.VListItem(
                v_for="item in sweep_config_palette",
                key="item.value",
                click=(self._on_palette_config, "[item.value]"),
                title=("item.title",),
                subtitle=("item.owner ? 'model: ' + item.owner : ''",),
            ):
                with v3.Template(v_slot_append=True):
                    v3.VIcon(
                        icon=(
                            "sweep_dims_on_canvas.includes(item.value)"
                            " ? 'mdi-minus-circle-outline'"
                            " : 'mdi-plus-circle-outline'",
                        ),
                        color=(
                            "sweep_dims_on_canvas.includes(item.value)"
                            " ? 'error' : 'primary'",
                        ),
                        size="small",
                    )
            v3.VDivider(classes="my-2")
            v3.VListSubheader("Fields / boundary conditions")
            with v3.VListItem(
                v_for="item in sweep_field_palette",
                key="item.value",
                click=(self._on_palette_field, "[item.value]"),
                title=("item.title",),
                subtitle=("item.owner ? 'model: ' + item.owner : 'per-case'",),
            ):
                with v3.Template(v_slot_append=True):
                    v3.VIcon(
                        icon=(
                            "sweep_field_on_canvas.includes(item.value)"
                            " ? 'mdi-minus-circle-outline'"
                            " : 'mdi-plus-circle-outline'",
                        ),
                        color=(
                            "sweep_field_on_canvas.includes(item.value)"
                            " ? 'error' : 'primary'",
                        ),
                        size="small",
                    )
            v3.VDivider(classes="my-2")
            v3.VListSubheader("Mesh dimension")
            with v3.VListItem(
                v_for="item in sweep_mesh_palette",
                key="item.value",
                click=(self._on_palette_mesh, "[item.value]"),
                title=("item.title",),
                subtitle="re-meshes per variant",
            ):
                with v3.Template(v_slot_append=True):
                    v3.VIcon(
                        icon=(
                            "sweep_mesh_on_canvas.includes(item.value)"
                            " ? 'mdi-minus-circle-outline'"
                            " : 'mdi-plus-circle-outline'",
                        ),
                        color=(
                            "sweep_mesh_on_canvas.includes(item.value)"
                            " ? 'error' : 'primary'",
                        ),
                        size="small",
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
                        update_modelValue=(self.toggle_rule, "[r.name, $event]"),
                    )

    def _dim_node_template(
        self, html: Any, v3: Any, handle: Any, custom_node: Any
    ) -> None:
        """A dimension node: a compact overview chip (Phase L).

        The config's form no longer lives in the node — it is edited in the
        Configure tab below the canvas. The node shows only the config title,
        its variant count and the ``cfg:`` source handle; clicking it selects
        the dimension for the Configure tab.
        """
        with custom_node("dim"), html.Div(classes="nf-sweep-node"):
            with html.Div(
                "{{ props.data.label }}", classes="nf-sweep-header nf-sweep-header-dim"
            ):
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

    def _rule_node_template(self, html: Any, handle: Any, custom_node: Any) -> None:
        """A fixed pipeline node: config ports per dimension + typed file ports."""
        with custom_node("rule"), html.Div(classes="nf-sweep-node"):
            html.Div(
                "{{ props.data.label }}", classes="nf-sweep-header nf-sweep-header-rule"
            )
            with html.Div(
                v_for="dim in props.data.cfg_dims", key=("dim",), classes="port-row"
            ):
                handle(type="target", position="left", id=("'cfg:' + dim",))
                html.Span("{{ dim }}")
            with html.Div(
                v_for="f in props.data.inputs", key=("f",), classes="port-row"
            ):
                handle(
                    type="target",
                    position="left",
                    id=("'in:' + f",),
                    classes="nf-handle-file",
                )
                html.Span("{{ f.split('/').pop() }}")
            with html.Div(
                v_for="f in props.data.outputs", key=("f",), classes="port-row out"
            ):
                html.Span("{{ f.split('/').pop() }}")
                handle(
                    type="source",
                    position="right",
                    id=("'out:' + f",),
                    classes="nf-handle-file",
                )
