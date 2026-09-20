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
  packaged rule library (:mod:`neofoam.tooling.workflow.rules`). A node's *series
  generator* creates variants from a value list or a linear/log range,
  auto-named like ``nu1e-05``.
- Below the canvas, a tabbed card holds the **parameters table** — every
  combination, one row per case, with the varied parameters' concrete values —
  and the **Snakemake graph** of the exported workflow (``snakemake --dag`` /
  ``--rulegraph``, rendered via :mod:`neofoam.tooling.workflow.dag`).
- Right-clicking a canvas node opens a context menu: remove the dimension /
  disable the rule (required rules are locked).

Only this module and its layout (:mod:`neofoam.ui.sweep_view`) talk to trame; the
canvas data model, the export and the DAG rendering live UI-free in
:mod:`neofoam.tooling.workflow`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from neofoam.io.pydantic_schema import slice_schema
from neofoam.tooling.workflow.dag import dag_graph
from neofoam.tooling.workflow.rules import DEFAULT_ENABLED, MESH_DIM, default_registry
from neofoam.tooling.workflow.sweep import (
    SweepDimension,
    autowire,
    dim_node,
    export_sweep,
    load_sweep,
    rule_nodes,
)
from neofoam.tooling.workflow.sweep_runner import config_classes_by_name
from neofoam.ui import sweep_view
from neofoam.ui._paths import _resolve_target
from neofoam.ui.case_load import apply_configs_to_forms, read_case_configs
from neofoam.ui.forms import FormEntry, build_field_forms, build_mesh_forms
from neofoam.ui.steps import build_model_families, selection_key
from neofoam.ui.sweep_model import DimensionState, SweepModel, series_values

#: Above this many cases the count chip turns warning-colored and Export asks
#: for confirmation (a factorized sweep grows fast — a guard against a typo in
#: a series generator spawning thousands of runs).
_CASE_WARN_THRESHOLD = 64

#: The reserved CAD geometry axis — no solver config backs it, so it is exported
#: and restored separately from the config dimensions (``MESH_DIM`` is upstream).
_CAD_DIM = "cad"

#: Hard bound on one ``snakemake --dag`` run (it grows with the case count:
#: ~1 s for a single case, ~2 s at 200). Past this the graph is reported as
#: failed rather than leaving the button spinning forever.
_DAG_TIMEOUT_S = 120.0

__all__ = ["SweepPanel"]

#: Rules every pipeline needs (mirrors ``RuleRegistry.plan``) — locked in the
#: palette so they cannot be toggled off.
_REQUIRED_RULES = frozenset({"all", "setup", "solve", "setup_mesh"})


def _dim_name(node_id: str) -> str:
    """The model dimension name behind a ``dim:<name>`` canvas node id."""
    return node_id[len("dim:") :] if node_id.startswith("dim:") else node_id


class SweepPanel:
    """State, controllers and canvas of the parameter-sweep step."""

    def __init__(self, server: Any, entries: list[FormEntry], solver: Any, solver_name: str):
        self._server = server
        self._solver_name = solver_name
        # Kept whole for the base-case restore (Load reopens the study): reading a
        # case back in fills every form entry, not just the sweepable ones.
        self._solver = solver
        self._entries = entries
        self._families = build_model_families(solver)
        # Sweepable dimensions: dict-kind configs only (field halves are merged
        # at save time and their schemas mutate with the geometry scan).
        self._dims: dict[str, FormEntry] = {e.config_name: e for e in entries if e.kind == "dict"}
        # Sweepable mesh dicts (blockMesh/snappy) — sources of the keyed ``mesh``
        # dimension, kept out of the physics wizard (see forms._MESH_FILES).
        self._mesh_dims: dict[str, FormEntry] = {e.config_name: e for e in build_mesh_forms(solver)}
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
        self._seed_state()
        self._register_controllers()

    def _seed_state(self) -> None:
        """Every ``sweep_*`` state variable at its initial value."""
        state = self._server.state
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
            {"title": e.title, "value": name} for name, e in sorted(self._mesh_dims.items())
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
        # V4 load: the base case a load would restore over the current forms, and
        # the dialog asking whether it may.
        state.sweep_load_confirm_show = False
        state.sweep_load_base = ""
        # V3 staleness: the canvas differs from the last export.
        state.sweep_dirty = False
        state.sweep_headers = []
        state.sweep_rows = []
        state.sweep_dag_mode = "dag"
        state.sweep_dag_error = ""
        state.sweep_dag_busy = False
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

    def _register_controllers(self) -> None:
        """The ``ctrl.sweep_*`` seams, the JS triggers and the state watchers."""
        server = self._server
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
        ctrl.sweep_confirm_load = self.confirm_load
        ctrl.sweep_refresh_dag = self.refresh_dag
        ctrl.sweep_get_nodes = lambda: list(self._editor.nodes) if self._editor else []
        ctrl.sweep_get_dag_nodes = lambda: list(self._dag_view.nodes) if self._dag_view else []
        # The variant ops are node-template event handlers; registering them on
        # the controller too gives headless drivers (tests, scripted examples)
        # the same seam the browser uses.
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
        # A CAD axis feeds the mesh chain; it is not a per-case setup config port.
        config_dims = [s.name for s in states if s.kind != "cad"]
        rules = rule_nodes(config_dims, enabled=self._model.enabled)
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

        # Holds a strong ref — asyncio only weakly references pending tasks.
        self._fit_task = loop.create_task(_fit())

    # -- canvas actions --------------------------------------------------------

    def _after_add(self, dim_name: str, message: str) -> None:
        """Re-derive the canvas after an add and configure the new dimension."""
        self._rebuild_graph()
        self._refresh_overview()
        # Configure the freshly added dimension straight away.
        self._server.state.sweep_cfg_dim = dim_name
        self._sync_cfg_mirror()
        self._editor.fit_view()  # bring the new node into the viewport
        self._notify(message=message)

    def _after_remove(self, message: str) -> None:
        """Re-derive the canvas after a removal and re-frame what remains."""
        self._rebuild_graph()
        self._refresh_overview()
        self._sync_cfg_mirror()
        self._fit_soon(self._editor)  # re-frame the canvas around what remains
        self._notify(message=message)

    def _owner_model_blocked(self, entry: FormEntry) -> bool:
        """True (after notifying) when the entry's optional model is not selected."""
        owner = entry.owner_model
        if owner is None or self._server.state[selection_key(owner)]:
            return False
        self._notify(
            error=f"'{entry.title}' belongs to the unselected model"
            f" '{owner}' — enable it in the Models step first."
        )
        return True

    def add_dimension(self, name: str | None = None, fields: list[str] | None = None) -> None:
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
        if self._owner_model_blocked(entry):
            return
        if self._model.has(entry.config_name):
            self._notify(error=f"'{entry.title}' is already on the canvas.")
            return
        schema = entry.schema
        if fields:
            known = set(schema.get("properties", {}))
            unknown = [f for f in fields if f not in known]
            if unknown:
                self._notify(error=f"unknown parameter(s) {unknown} for '{entry.title}'")
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
        self._after_add(entry.config_name, f"Added dimension '{entry.title}'.")

    def add_cad_dimension(self, model_path: str, params: dict[str, float]) -> None:
        """Add the CAD geometry axis (reserved dim ``cad``) to the canvas.

        Called by the FoamCAD step's "Add to sweep": ``params`` are the selected
        parametric aliases and their current values. The CAD dimension is an
        independent, STL-producing axis (it composes with the mesh axis as
        ``cad × mesh``) and is configured as a numeric parameter map.
        """
        if self._model.has(_CAD_DIM):
            self._notify(error="A CAD dimension is already on the canvas.")
            return
        self._model.add_cad_dimension(
            _CAD_DIM, title="CAD geometry", model_path=model_path, params=params
        )
        self._after_add(_CAD_DIM, "Added CAD dimension.")

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
        self._after_add(MESH_DIM, f"Added mesh source '{entry.title}'.")

    def add_field_source(self, config_name: str) -> None:
        """Add a whole ``0/<field>`` config as a (per-case) sweep dimension.

        A field config (e.g. ``u_field_config``) is a regular dimension — it
        applies in the per-case clone via ``setup.json``, not the keyed mesh
        mini-case — so it goes through the normal :meth:`SweepModel.add_dimension`
        path, seeded from the saved base case. Gated on its owner model like any
        optional-model config (``t_field_config`` needs Boussinesq enabled).
        """
        entry = self._field_dims.get(config_name)
        if entry is None:
            self._notify(error=f"Unknown field '{config_name}'.")
            return
        if self._owner_model_blocked(entry):
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
        self._after_add(config_name, f"Added field dimension '{entry.title}'.")

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
        title = entry.title if entry else config_name
        self._after_remove(f"Removed mesh source '{title}'.")

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
        if self._owner_model_blocked(entry):
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
        self.add_dimension(state.sweep_pick_config, fields=list(state.sweep_pick_selected) or None)

    def remove_dimension(self, name: str) -> None:
        """Remove a dimension from the canvas (its variants are dropped)."""
        entry = self._dims.get(name)
        try:
            self._model.remove_dimension(name)
        except ValueError:
            self._notify(error=f"'{name}' is not on the canvas.")
            return
        title = entry.title if entry else name
        self._after_remove(f"Removed dimension '{title}'.")

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

    def load_exported(self, force: bool = False) -> None:
        """V4: reopen an exported sweep — its canvas *and* its base case.

        Rebuilds the dimension nodes (full variant payloads), restores the
        enabled rules and re-derives the pipeline, and reads the sidecar's base
        case back into the wizard forms — so a prior study can be reopened,
        re-validated (V1 badges anything that no longer validates) and extended.
        A sweep exported from a different solver is refused whole (this wizard is
        built around one solver's configs); dimensions with no matching config on
        this solver are reported and skipped.

        Restoring the base case overwrites the current forms, so the first call
        stops to ask; ``force=True`` (the confirm dialog's action) goes ahead.
        """
        state = self._server.state
        try:
            out_dir = self._out_dir()
            loaded = load_sweep(out_dir)
        except (ValueError, OSError) as exc:
            self._notify(error=f"Could not load sweep: {exc}")
            return
        if loaded.solver_name != self._solver_name:
            self._notify(
                error=f"That sweep was exported from solver '{loaded.solver_name}',"
                f" but this wizard is running '{self._solver_name}' — nothing was"
                f" loaded. Restart the wizard on '{loaded.solver_name}' to reopen it."
            )
            return
        if loaded.base_case and not force:
            state.sweep_load_base = loaded.base_case
            state.sweep_load_confirm_show = True
            return
        if loaded.base_case:
            self._restore_base_case(loaded.base_case)

        states, skipped = self._loaded_dimensions(loaded.dimensions)

        # Restore the enabled rules (fall back to the defaults if the sidecar's
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
        if loaded.base_case:
            note += f", base case {loaded.base_case} back in the forms"
        if skipped:
            note += (
                " — skipped unsupported dimension(s), backed by no"
                f" '{self._solver_name}' config: {', '.join(skipped)}"
            )
        self._notify(message=note, error="")

    def _loaded_dimensions(
        self, dimensions: dict[str, dict[str, dict[str, Any]]]
    ) -> tuple[list[DimensionState], list[str]]:
        """The canvas dimensions an exported sweep restores to, plus the skipped ones."""
        states: list[DimensionState] = []
        skipped: list[str] = []
        for dim, variants in sorted(dimensions.items()):
            if dim in (_CAD_DIM, MESH_DIM):
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
        mesh_variants = dimensions.get(MESH_DIM)
        if mesh_variants:
            sources = sorted({name for payload in mesh_variants.values() for name in payload})
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
                    name=MESH_DIM,
                    title="Mesh",
                    schema={"type": "object", "properties": mesh_props},
                    entries={k: dict(v) for k, v in mesh_variants.items()},
                    selected=next(iter(mesh_variants)),
                    kind="mesh",
                )
            )

        # A CAD axis is not restored here: neofoam.tooling.workflow's LoadedSweep
        # carries no cad model path, so the axis cannot be rebuilt. It comes back
        # with the CAD plugin, which owns both halves of that round-trip.
        return states, skipped

    def _restore_base_case(self, base_case: str) -> None:
        """Reopen the sweep's base case: the target field and every wizard form.

        Without this the loaded axes would sit on top of whatever the wizard
        happened to hold, and Export would clone the wrong case.
        """
        state = self._server.state
        state.target_dir = base_case
        configs = read_case_configs(Path(base_case), self._solver)
        apply_configs_to_forms(state, self._entries, self._families, configs, Path(base_case))

    def confirm_load(self) -> None:
        """The load dialog's "Load anyway" action — the base case replaces the forms."""
        self._server.state.sweep_load_confirm_show = False
        self.load_exported(force=True)

    def _out_dir(self) -> Path:
        """The sweep directory field, else ``<target>-sweep`` — always absolute.

        Raises ``ValueError`` when the field it is derived from is blank or
        relative; otherwise the default ``<target>-sweep`` becomes a directory
        literally named ``-sweep`` under the server's launch directory.
        """
        state = self._server.state
        if state.sweep_out_dir:
            return _resolve_target(state.sweep_out_dir, "sweep directory")
        base = _resolve_target(state.target_dir, "target directory")
        return base.with_name(base.name + "-sweep")

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
        try:
            base_case = _resolve_target(state.target_dir, "target directory")
            out_dir = self._out_dir()
        except ValueError as exc:
            self._notify(error=str(exc))
            return
        # A CAD axis is a plugin concern: neofoam.tooling.workflow.sweep exports the
        # solver's own dimensions only. Refuse loudly rather than dropping the axis.
        if self._model.cad_dimensions():
            self._notify(
                error="CAD sweep dimensions need the CAD plugin — the packaged "
                "workflow layer cannot export a cad axis. Remove the CAD "
                "dimension or install the plugin."
            )
            return
        try:
            result = export_sweep(
                out_dir,
                solver_name=self._solver_name,
                base_case=base_case,
                dimensions=self._model.to_dimensions(),
                classes=self._classes,
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

    async def refresh_dag(self) -> None:
        """Export the sweep, run ``snakemake --dag|--rulegraph`` and render it."""
        state = self._server.state
        # The button is disabled while busy, but a queued click still lands here:
        # a second run would export and shell out to snakemake all over again.
        if state.sweep_dag_busy:
            return
        self.export()
        # Nothing was written if export errored or is awaiting confirmation.
        if state.sweep_error or state.sweep_confirm_show:
            return
        with state:  # flush now — the spinner has to show before we await
            state.sweep_dag_busy = True
        try:
            await self._render_dag(state.sweep_out_dir, state.sweep_dag_mode)
        finally:
            # Also flushes whatever _render_dag wrote: this may run as a detached
            # task (the mode listener), outside any state context of its own.
            with state:
                state.sweep_dag_busy = False

    async def _render_dag(self, out_dir: str, mode: str) -> None:
        """Run snakemake off the loop and put the resulting graph on the canvas."""
        state = self._server.state
        try:
            # snakemake blocks for a second or more; a worker thread keeps the
            # loop ticking, and every state write below stays on the loop.
            nodes, edges = await asyncio.wait_for(
                asyncio.to_thread(dag_graph, out_dir, mode), _DAG_TIMEOUT_S
            )
        except asyncio.TimeoutError:  # not the builtin before 3.11
            self._fail_dag(f"snakemake --{mode} did not finish within {_DAG_TIMEOUT_S:.0f}s.")
            return
        except (RuntimeError, ValueError) as exc:
            self._fail_dag(str(exc))
            return
        state.sweep_dag_error = ""
        self._dag_generated = True
        self._dag_view.graph = {"nodes": nodes, "edges": edges}
        self._fit_soon(self._dag_view)

    def _fail_dag(self, message: str) -> None:
        """Report a graph failure and leave no stale graph behind it."""
        self._server.state.sweep_dag_error = message
        self._dag_view.clear_graph()

    async def _on_dag_mode(self, sweep_dag_mode: str, **_: Any) -> None:
        if self._dag_generated:
            await self.refresh_dag()

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
                error=f"'{dim.title}' has no numeric top-level parameter to generate a series for."
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
        count = self._model.generate_series(name, param, values, replace=state.sweep_gen_replace)
        self._write_dim_node(name, echo=True)
        self._refresh_overview()
        self._sync_cfg_mirror()
        state.sweep_gen_error = ""
        state.sweep_gen_show = False
        self._notify(
            message=f"Generated {count} '{param}' variant(s) on '{self._model.dims[name].title}'."
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
        self._editor, self._dag_view = sweep_view.render(self, json_forms)
        seed_nodes = rule_nodes([], enabled=self._model.enabled)
        self._editor.graph = {"nodes": seed_nodes, "edges": autowire(seed_nodes)}
