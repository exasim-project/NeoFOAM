# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for the Parameters step (sweep canvas + export controller)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")
pytest.importorskip("trame")
pytest.importorskip("trame_flow")

from trame.app import get_server  # noqa: E402

from neofoam.mcp import tools  # noqa: E402
from neofoam.mcp.registry import resolve_solver  # noqa: E402
from neofoam.ui import build_app  # noqa: E402


def _dim_nodes(server):
    return [n for n in server.controller.sweep_get_nodes() if n["type"] == "dim"]


def _seed_defaults(server) -> None:
    """The minimal valid form state used by the save round-trip app tests."""
    solver = resolve_solver("incompressibleFluid")
    defaults = tools.config_schema(solver, "transport_properties_config").defaults
    for entry in server.controller.get_entries():
        server.state[entry.state_key] = (
            dict(defaults) if entry.config_name == "transport_properties_config" else {}
        )


def test_sweep_state_and_canvas_seeded():
    server = build_app(server=get_server("neofoam_ui_test_sweep_seed"))
    state = server.state
    assert state.sweep_case_count == 0
    assert state.sweep_exported == []
    # Choices are dict-kind configs only (no field_in/field_bc halves).
    values = {c["value"] for c in state.sweep_dim_choices}
    assert "transport_properties_config" in values
    assert "control_dict_config" in values
    assert not any(v.startswith("field") for v in values)
    entries = {
        e.config_name: e for e in server.controller.get_entries() if e.kind == "dict"
    }
    assert values == set(entries)
    # The default rule-library pipeline is on the canvas, pre-wired.
    nodes = server.controller.sweep_get_nodes()
    assert [n["data"]["rule"] for n in nodes if n["type"] == "rule"] == [
        "setup_mesh",
        "blockMesh",
        "snappyHexMesh",
        "checkMesh",
        "setup",
        "solve",
        "all",
    ]
    assert _dim_nodes(server) == []


def test_add_dimension_seeds_from_live_form_state():
    server = build_app(server=get_server("neofoam_ui_test_sweep_add"))
    state, ctrl = server.state, server.controller
    entry = next(
        e
        for e in ctrl.get_entries()
        if e.config_name == "transport_properties_config" and e.kind == "dict"
    )
    state[entry.state_key] = {"transportModel": "Newtonian", "nu": 3e-5}

    state.sweep_dim_pick = "transport_properties_config"
    ctrl.sweep_add_dimension()

    (node,) = _dim_nodes(server)
    assert node["id"] == "dim:transport_properties_config"
    assert node["data"]["entries"] == {
        "base": {"transportModel": "Newtonian", "nu": 3e-5}
    }
    assert node["data"]["schema"] == entry.schema
    # setup's config ports and the wiring follow.
    setup = next(n for n in ctrl.sweep_get_nodes() if n["id"] == "rule:setup")
    assert setup["data"]["cfg_dims"] == ["transport_properties_config"]
    assert state.sweep_case_count == 1

    # Duplicate add is refused.
    ctrl.sweep_add_dimension()
    assert len(_dim_nodes(server)) == 1
    assert "already on the canvas" in state.sweep_error


def test_add_dimension_requires_selected_owner_model():
    server = build_app(server=get_server("neofoam_ui_test_sweep_owner"))
    state, ctrl = server.state, server.controller
    owned = next(
        (
            e
            for e in ctrl.get_entries()
            if e.kind == "dict" and e.owner_model is not None
        ),
        None,
    )
    if owned is None:
        pytest.skip("solver has no optional-model-owned dict configs")
    state.sweep_dim_pick = owned.config_name
    ctrl.sweep_add_dimension()
    assert _dim_nodes(server) == []
    assert owned.owner_model in state.sweep_error

    state[f"sel_{owned.owner_model}"] = True
    ctrl.sweep_add_dimension()
    assert len(_dim_nodes(server)) == 1


def test_variant_add_rename_delete():
    server = build_app(server=get_server("neofoam_ui_test_sweep_variants"))
    state, ctrl = server.state, server.controller
    state.sweep_dim_pick = "transport_properties_config"
    ctrl.sweep_add_dimension()
    node_id = "dim:transport_properties_config"

    # Add clones the selected variant.
    ctrl.sweep_variant_add(node_id)
    (node,) = _dim_nodes(server)
    assert set(node["data"]["entries"]) == {"base", "variant-1"}
    assert node["data"]["selected"] == "variant-1"
    assert state.sweep_case_count == 2
    # The node seam keeps the Configure mirror in sync on the active dimension
    # (the dimension was added via the palette, so it is the active one).
    assert set(state.sweep_cfg_variants) == {"base", "variant-1"}
    assert state.sweep_cfg_selected == "variant-1"
    assert state.sweep_cfg_data == node["data"]["entries"]["variant-1"]

    # Rename validates the name.
    ctrl.sweep_rename_buffer(node_id, "bad name!")
    ctrl.sweep_variant_rename(node_id)
    assert "Invalid variant name" in state.sweep_error
    ctrl.sweep_rename_buffer(node_id, "nu2e-05")
    ctrl.sweep_variant_rename(node_id)
    (node,) = _dim_nodes(server)
    assert set(node["data"]["entries"]) == {"base", "nu2e-05"}
    # The mirror follows the rename.
    assert state.sweep_cfg_selected == "nu2e-05"
    assert "nu2e-05" in state.sweep_cfg_variants

    # Field edits land on the selected variant (the trigger's Python side).
    ctrl.sweep_variant_edit(node_id, {"transportModel": "Newtonian", "nu": 9e-5})
    (node,) = _dim_nodes(server)
    assert node["data"]["entries"]["nu2e-05"]["nu"] == 9e-5

    # Delete keeps at least one variant.
    ctrl.sweep_variant_delete(node_id)
    (node,) = _dim_nodes(server)
    assert set(node["data"]["entries"]) == {"base"}
    # The mirror never points at the deleted variant — it falls back to a
    # surviving one.
    assert state.sweep_cfg_selected in node["data"]["entries"]
    assert state.sweep_cfg_selected != "nu2e-05"
    assert set(state.sweep_cfg_variants) == set(node["data"]["entries"])
    ctrl.sweep_variant_delete(node_id)
    assert set(node["data"]["entries"]) == {"base"}
    assert "at least one variant" in state.sweep_error


def test_configure_tab_mirrors_active_dimension():
    # Phase L: the forms live in the Configure tab; adding a dimension makes it
    # the active one and mirrors its live node data into the sweep_cfg_* state.
    server = build_app(server=get_server("neofoam_ui_test_sweep_cfg_mirror"))
    state, ctrl = server.state, server.controller
    assert state.sweep_tab == "configure"

    entry = next(
        e
        for e in ctrl.get_entries()
        if e.config_name == "transport_properties_config" and e.kind == "dict"
    )
    state[entry.state_key] = {"transportModel": "Newtonian", "nu": 1e-5}
    state.sweep_dim_pick = "transport_properties_config"
    ctrl.sweep_add_dimension()

    assert state.sweep_cfg_dim == "transport_properties_config"
    assert [c["value"] for c in state.sweep_cfg_chips] == [
        "transport_properties_config"
    ]
    assert state.sweep_cfg_chips[0]["count"] == 1
    assert state.sweep_cfg_variants == ["base"]
    assert state.sweep_cfg_selected == "base"
    assert state.sweep_cfg_data == {"transportModel": "Newtonian", "nu": 1e-5}
    assert "nu" in state.sweep_cfg_schema.get("properties", {})


def test_configure_tab_variant_ops_and_selection():
    server = build_app(server=get_server("neofoam_ui_test_sweep_cfg_ops"))
    state, ctrl = server.state, server.controller
    entry = next(
        e
        for e in ctrl.get_entries()
        if e.config_name == "transport_properties_config" and e.kind == "dict"
    )
    # Give base a distinct payload so a re-push is observable.
    state[entry.state_key] = {"transportModel": "Newtonian", "nu": 1e-5}
    state.sweep_dim_pick = "transport_properties_config"
    ctrl.sweep_add_dimension()

    # Variant ops via the tab seams operate on the active dimension.
    ctrl.sweep_cfg_variant_add()
    assert set(state.sweep_cfg_variants) == {"base", "variant-1"}
    assert state.sweep_cfg_selected == "variant-1"

    ctrl.sweep_cfg_rename_buffer("nu2e-05")
    ctrl.sweep_cfg_variant_rename()
    assert set(state.sweep_cfg_variants) == {"base", "nu2e-05"}
    assert state.sweep_cfg_selected == "nu2e-05"

    # Edit the selected clone; base must keep its own payload.
    ctrl.sweep_variant_edit(
        "dim:transport_properties_config", {"transportModel": "Newtonian", "nu": 2e-5}
    )

    # Switching the selected variant re-pushes THAT variant's payload (base's,
    # not the edited clone's).
    ctrl.sweep_cfg_variant_select("base")
    assert state.sweep_cfg_selected == "base"
    assert state.sweep_cfg_data == {"transportModel": "Newtonian", "nu": 1e-5}

    ctrl.sweep_cfg_variant_delete()
    assert state.sweep_cfg_variants == ["nu2e-05"]


def test_configure_tab_switches_between_and_drops_dimensions():
    server = build_app(server=get_server("neofoam_ui_test_sweep_cfg_switch"))
    state, ctrl = server.state, server.controller
    ctrl.sweep_toggle_dimension("transport_properties_config")
    ctrl.sweep_toggle_dimension("control_dict_config")
    assert {c["value"] for c in state.sweep_cfg_chips} == {
        "transport_properties_config",
        "control_dict_config",
    }

    # A canvas click on a node configures it (and jumps to the tab).
    state.sweep_tab = "table"
    ctrl.sweep_select_cfg("control_dict_config")
    assert state.sweep_cfg_dim == "control_dict_config"

    # Removing the active dimension falls back to the remaining one.
    ctrl.sweep_remove_dimension("control_dict_config")
    assert state.sweep_cfg_dim == "transport_properties_config"

    # Removing the last dimension clears the tab.
    ctrl.sweep_remove_dimension("transport_properties_config")
    assert state.sweep_cfg_dim == ""
    assert state.sweep_cfg_chips == []
    assert state.sweep_cfg_data == {}


def test_live_validation_badges_and_blocks_export():
    # V1: an invalid variant badges its node, fills the table's validation
    # column, and blocks Export.
    server = build_app(server=get_server("neofoam_ui_test_sweep_v1"))
    state, ctrl = server.state, server.controller
    entry = next(
        e
        for e in ctrl.get_entries()
        if e.config_name == "transport_properties_config" and e.kind == "dict"
    )
    state[entry.state_key] = {"transportModel": "Newtonian", "nu": 1e-5}
    ctrl.sweep_toggle_dimension("transport_properties_config")
    node_id = "dim:transport_properties_config"
    assert state.sweep_valid
    assert state.sweep_rows[0]["validation"] == "✓"

    # An out-of-range nu fails validation.
    ctrl.sweep_variant_edit(node_id, {"transportModel": "Newtonian", "nu": "abc"})
    assert not state.sweep_valid
    (node,) = _dim_nodes(server)
    assert node["data"]["error"]  # badge message set
    assert state.sweep_rows[0]["validation"] != "✓"
    # Export refuses while invalid.
    state.scaffolded = ["1"]
    ctrl.sweep_export()
    assert "invalid" in state.sweep_error
    assert state.sweep_exported == []

    # Fixing it clears the gate.
    ctrl.sweep_variant_edit(node_id, {"transportModel": "Newtonian", "nu": 1e-5})
    assert state.sweep_valid
    (node,) = _dim_nodes(server)
    assert node["data"]["error"] == ""


def test_configure_tab_surfaces_selected_variant_error():
    server = build_app(server=get_server("neofoam_ui_test_sweep_v1_cfg"))
    state, ctrl = server.state, server.controller
    ctrl.sweep_toggle_dimension("transport_properties_config")
    node_id = "dim:transport_properties_config"
    ctrl.sweep_variant_edit(node_id, {"transportModel": "Newtonian", "nu": "abc"})
    # The active variant's message is mirrored for the Configure tab alert.
    assert state.sweep_cfg_dim == "transport_properties_config"
    assert state.sweep_cfg_error


def test_case_count_factorization_and_threshold_guard(tmp_path, monkeypatch):
    # V2: the chip label factorizes; over the cap Export asks to confirm.
    import neofoam.ui.sweep_panel as sp

    monkeypatch.setattr(sp, "_CASE_WARN_THRESHOLD", 3)
    server = build_app(server=get_server("neofoam_ui_test_sweep_v2"))
    state, ctrl = server.state, server.controller
    _seed_defaults(server)
    state.target_dir = str(tmp_path / "base")
    ctrl.save_case()

    ctrl.sweep_toggle_dimension("transport_properties_config")
    node_id = "dim:transport_properties_config"
    for _ in range(3):
        ctrl.sweep_variant_add(node_id)
    # 4 variants of one dimension → "4 case(s)", over the cap of 3.
    assert state.sweep_count_label == "4 case(s)"
    assert state.sweep_count_warn

    ctrl.sweep_export()
    # Over the cap: no export yet, the confirm dialog is open.
    assert state.sweep_confirm_show
    assert state.sweep_exported == []
    # Confirming exports.
    ctrl.sweep_confirm_export()
    assert not state.sweep_confirm_show
    assert state.sweep_exported

    # A second dimension factorizes the label.
    ctrl.sweep_toggle_dimension("control_dict_config")
    assert "×" in state.sweep_count_label
    assert state.sweep_count_label.endswith("case(s)")


def test_staleness_flag_set_on_edit_cleared_on_export(tmp_path):
    # V3: any canvas change marks the sweep dirty; export clears it.
    server = build_app(server=get_server("neofoam_ui_test_sweep_v3"))
    state, ctrl = server.state, server.controller
    _seed_defaults(server)
    state.target_dir = str(tmp_path / "base")
    ctrl.save_case()
    assert not state.sweep_dirty

    ctrl.sweep_toggle_dimension("transport_properties_config")
    assert state.sweep_dirty

    ctrl.sweep_export()
    assert state.sweep_exported
    assert not state.sweep_dirty

    # A further edit re-dirties.
    ctrl.sweep_variant_add("dim:transport_properties_config")
    assert state.sweep_dirty


def test_palette_toggle_dimension_adds_and_removes():
    server = build_app(server=get_server("neofoam_ui_test_sweep_palette"))
    state, ctrl = server.state, server.controller
    # The palette lists the same dict-kind configs as the (kept) choices list.
    palette = {p["value"] for p in state.sweep_config_palette}
    assert palette == {c["value"] for c in state.sweep_dim_choices}

    ctrl.sweep_toggle_dimension("transport_properties_config")
    assert len(_dim_nodes(server)) == 1
    assert state.sweep_dims_on_canvas == ["transport_properties_config"]

    ctrl.sweep_toggle_dimension("transport_properties_config")
    assert _dim_nodes(server) == []
    assert state.sweep_dims_on_canvas == []
    assert "Removed dimension" in state.sweep_status


def test_dim_picker_slices_display_schema_but_keeps_full_payloads():
    server = build_app(server=get_server("neofoam_ui_test_sweep_picker"))
    state, ctrl = server.state, server.controller
    entry = next(
        e
        for e in ctrl.get_entries()
        if e.config_name == "transport_properties_config" and e.kind == "dict"
    )
    state[entry.state_key] = {"transportModel": "Newtonian", "nu": 1e-5}

    ctrl.sweep_open_dim_picker("transport_properties_config")
    assert state.sweep_pick_show
    assert {o["value"] for o in state.sweep_pick_options} == {"transportModel", "nu"}

    state.sweep_pick_selected = ["nu"]
    ctrl.sweep_confirm_dim()
    assert not state.sweep_pick_show
    (node,) = _dim_nodes(server)
    # The node renders only the picked parameter …
    assert list(node["data"]["schema"]["properties"]) == ["nu"]
    assert node["data"]["fields"] == ["nu"]
    # … but the variant payload stays full (export validates whole configs).
    assert node["data"]["entries"]["base"] == {
        "transportModel": "Newtonian",
        "nu": 1e-5,
    }

    # A sliced form edit merges — the unrendered keys survive.
    ctrl.sweep_variant_edit("dim:transport_properties_config", {"nu": 5e-5})
    (node,) = _dim_nodes(server)
    assert node["data"]["entries"]["base"] == {
        "transportModel": "Newtonian",
        "nu": 5e-5,
    }

    # Empty selection = the whole form (today's behavior).
    ctrl.sweep_remove_dimension("transport_properties_config")
    ctrl.sweep_open_dim_picker("transport_properties_config")
    ctrl.sweep_confirm_dim()
    (node,) = _dim_nodes(server)
    assert "transportModel" in node["data"]["schema"]["properties"]
    assert node["data"]["fields"] == []


def test_dim_picker_respects_owner_model_gate():
    server = build_app(server=get_server("neofoam_ui_test_sweep_picker_owner"))
    state, ctrl = server.state, server.controller
    owned = next(
        (e for e in ctrl.get_entries() if e.kind == "dict" and e.owner_model),
        None,
    )
    if owned is None:
        pytest.skip("solver has no optional-model-owned dict configs")
    ctrl.sweep_open_dim_picker(owned.config_name)
    assert not state.sweep_pick_show
    assert owned.owner_model in state.sweep_error


def test_generate_variants_list_linear_log_and_replace():
    server = build_app(server=get_server("neofoam_ui_test_sweep_gen"))
    state, ctrl = server.state, server.controller
    entry = next(
        e
        for e in ctrl.get_entries()
        if e.config_name == "transport_properties_config" and e.kind == "dict"
    )
    state[entry.state_key] = {"transportModel": "Newtonian", "nu": 1e-5}
    ctrl.sweep_toggle_dimension("transport_properties_config")
    node_id = "dim:transport_properties_config"

    ctrl.sweep_open_generator(node_id)
    assert state.sweep_gen_show
    assert state.sweep_gen_param == "nu"  # the only numeric top-level parameter

    # Value-list mode appends auto-named variants with full payloads.
    state.sweep_gen_mode = "list"
    state.sweep_gen_values = "1e-5, 2e-5 4e-5"
    ctrl.sweep_generate_variants()
    assert not state.sweep_gen_show
    (node,) = _dim_nodes(server)
    assert set(node["data"]["entries"]) == {"base", "nu1e-05", "nu2e-05", "nu4e-05"}
    assert node["data"]["entries"]["nu2e-05"] == {
        "transportModel": "Newtonian",
        "nu": 2e-5,
    }
    assert state.sweep_case_count == 4

    # Linear range with replace drops the previous variants.
    ctrl.sweep_open_generator(node_id)
    state.sweep_gen_mode = "linear"
    state.sweep_gen_min, state.sweep_gen_max = "1", "3"
    state.sweep_gen_count = "3"
    state.sweep_gen_replace = True
    ctrl.sweep_generate_variants()
    (node,) = _dim_nodes(server)
    assert set(node["data"]["entries"]) == {"nu1", "nu2", "nu3"}
    assert node["data"]["entries"]["nu2"]["nu"] == 2.0
    assert state.sweep_case_count == 3

    # Log range; invalid input keeps the dialog open with the error.
    ctrl.sweep_open_generator(node_id)
    state.sweep_gen_mode = "log"
    state.sweep_gen_min, state.sweep_gen_max = "0", "1"
    state.sweep_gen_count = "3"
    state.sweep_gen_replace = False
    ctrl.sweep_generate_variants()
    assert state.sweep_gen_show
    assert "positive" in state.sweep_gen_error
    state.sweep_gen_min = "1e-6"
    state.sweep_gen_max = "1e-4"
    ctrl.sweep_generate_variants()
    (node,) = _dim_nodes(server)
    assert {"nu1e-06", "nu1e-05", "nu0.0001"} <= set(node["data"]["entries"])


def test_toggle_rule_rederives_pipeline_and_reverts_invalid():
    server = build_app(server=get_server("neofoam_ui_test_sweep_rules"))
    state, ctrl = server.state, server.controller

    def rule_names():
        return [
            n["data"]["rule"] for n in ctrl.sweep_get_nodes() if n["type"] == "rule"
        ]

    # Enabling the opt-in post rule inserts it before `all`.
    ctrl.sweep_toggle_rule("post", True)
    assert "post" in rule_names()
    post_row = next(r for r in state.sweep_rule_palette if r["name"] == "post")
    assert post_row["enabled"]
    # `all` now expands over post's output.
    all_node = next(n for n in ctrl.sweep_get_nodes() if n["id"] == "rule:all")
    assert all_node["data"]["inputs"] == ["cases/{case}/.post.done"]

    # Disabling a mid-chain mesh tool rewires the chain around it.
    ctrl.sweep_toggle_rule("snappyHexMesh", False)
    assert "snappyHexMesh" not in rule_names()
    check = next(n for n in ctrl.sweep_get_nodes() if n["id"] == "rule:checkMesh")
    assert check["data"]["inputs"] == ["meshes/{mesh}/.blockMesh.done"]

    # An invalid selection (no mesh creator) is refused and reverted.
    ctrl.sweep_toggle_rule("blockMesh", False)
    assert "blockMesh" in rule_names()
    assert "mesh" in state.sweep_error
    block_row = next(r for r in state.sweep_rule_palette if r["name"] == "blockMesh")
    assert block_row["enabled"]

    # Required rules are marked locked in the palette.
    required = {r["name"] for r in state.sweep_rule_palette if r["required"]}
    assert required == {"all", "setup", "solve", "setup_mesh"}


def test_parameters_table_lists_combinations_with_varied_values():
    server = build_app(server=get_server("neofoam_ui_test_sweep_table"))
    state, ctrl = server.state, server.controller
    assert state.sweep_rows == []

    entry = next(
        e
        for e in ctrl.get_entries()
        if e.config_name == "transport_properties_config" and e.kind == "dict"
    )
    state[entry.state_key] = {"transportModel": "Newtonian", "nu": 1e-5}
    ctrl.sweep_toggle_dimension("transport_properties_config")
    node_id = "dim:transport_properties_config"
    ctrl.sweep_variant_add(node_id)
    ctrl.sweep_rename_buffer(node_id, "nu2")
    ctrl.sweep_variant_rename(node_id)
    ctrl.sweep_variant_edit(node_id, {"transportModel": "Newtonian", "nu": 2e-5})

    # One column per varied parameter, titled by the bare parameter name
    # (params.yaml style) — transportModel is identical, so no column for it —
    # plus the trailing validation column (V1).
    assert [(h["title"], h["key"]) for h in state.sweep_headers] == [
        ("case", "case"),
        ("transportProperties", "transport_properties_config"),
        ("nu", "transport_properties_config__nu"),
        ("validation", "validation"),
    ]
    assert state.sweep_rows == [
        {
            "case": "base",
            "transport_properties_config": "base",
            "transport_properties_config__nu": 1e-5,
            "validation": "✓",
        },
        {
            "case": "nu2",
            "transport_properties_config": "nu2",
            "transport_properties_config__nu": 2e-5,
            "validation": "✓",
        },
    ]
    assert state.sweep_case_count == 2


def test_node_context_menu_removes_dim_and_disables_rule():
    server = build_app(server=get_server("neofoam_ui_test_sweep_menu"))
    state, ctrl = server.state, server.controller
    ctrl.sweep_toggle_dimension("transport_properties_config")

    # Right-click a dimension node → "Remove dimension", action removes it.
    ctrl.sweep_node_menu("dim:transport_properties_config", 100, 200)
    assert state.sweep_menu_show
    assert "Remove dimension" in state.sweep_menu_title
    assert not state.sweep_menu_locked
    ctrl.sweep_menu_action()
    assert not state.sweep_menu_show
    assert _dim_nodes(server) == []

    # Right-click an optional rule → "Disable rule", action disables it.
    ctrl.sweep_node_menu("rule:checkMesh", 0, 0)
    assert "Disable rule" in state.sweep_menu_title
    ctrl.sweep_menu_action()
    rules = [n["data"]["rule"] for n in ctrl.sweep_get_nodes() if n["type"] == "rule"]
    assert "checkMesh" not in rules

    # A required rule is locked — the action is a no-op.
    ctrl.sweep_node_menu("rule:solve", 0, 0)
    assert state.sweep_menu_locked
    ctrl.sweep_menu_action()
    rules = [n["data"]["rule"] for n in ctrl.sweep_get_nodes() if n["type"] == "rule"]
    assert "solve" in rules


def test_add_cad_dimension_puts_node_on_canvas_and_configures_it():
    server = build_app(server=get_server("neofoam_ui_test_sweep_cad"))
    state, ctrl = server.state, server.controller
    ctrl.sweep_add_cad_dimension("design.FCStd", {"tube_d": 8.0})

    (node,) = _dim_nodes(server)
    assert node["id"] == "dim:cad"
    assert node["data"]["entries"] == {"base": {"tube_d": 8.0}}
    # The freshly added CAD axis becomes the active Configure dimension.
    assert state.sweep_cfg_dim == "cad"
    # The CAD axis is NOT a per-case setup config port (it feeds the mesh chain).
    setup = next(n for n in ctrl.sweep_get_nodes() if n["id"] == "rule:setup")
    assert "cad" not in setup["data"]["cfg_dims"]
    assert state.sweep_case_count == 1

    # A second CAD dimension is refused (one reserved axis).
    ctrl.sweep_add_cad_dimension("design.FCStd", {"tube_d": 8.0})
    assert len(_dim_nodes(server)) == 1
    assert "already on the canvas" in state.sweep_error


def test_export_with_cad_dimension_writes_cad_configs(tmp_path):
    server = build_app(server=get_server("neofoam_ui_test_sweep_cad_export"))
    state, ctrl = server.state, server.controller
    _seed_defaults(server)
    state.target_dir = str(tmp_path / "base")
    ctrl.save_case()

    ctrl.sweep_toggle_dimension("transport_properties_config")
    ctrl.sweep_add_cad_dimension("geometry/design.FCStd", {"tube_d": 8.0})
    ctrl.sweep_export()

    assert state.sweep_error == ""
    out = Path(state.sweep_out_dir)
    cad_cfg = out / "configs" / "cad" / "base.json"
    assert cad_cfg.is_file()
    assert json.loads(cad_cfg.read_text()) == {"tube_d": 8.0}
    snake = (out / "Snakefile").read_text()
    assert 'CAD_MODEL = "geometry/design.FCStd"' in snake
    assert any(
        "cad_geometry.smk" in line
        for line in snake.splitlines()
        if line.startswith("include:")
    )


def test_export_requires_saved_case(tmp_path):
    server = build_app(server=get_server("neofoam_ui_test_sweep_nosave"))
    state, ctrl = server.state, server.controller
    state.sweep_dim_pick = "transport_properties_config"
    ctrl.sweep_add_dimension()
    state.target_dir = str(tmp_path / "base")
    ctrl.sweep_export()
    assert "Save the case first" in state.sweep_error
    assert not (tmp_path / "base-sweep").exists()


def test_export_writes_sweep_dir(tmp_path):
    server = build_app(server=get_server("neofoam_ui_test_sweep_export"))
    state, ctrl = server.state, server.controller
    _seed_defaults(server)
    state.target_dir = str(tmp_path / "base")
    ctrl.save_case()
    assert state.scaffolded

    state.sweep_dim_pick = "transport_properties_config"
    ctrl.sweep_add_dimension()
    node_id = "dim:transport_properties_config"
    ctrl.sweep_variant_add(node_id)
    ctrl.sweep_variant_edit(node_id, {"transportModel": "Newtonian", "nu": 2e-5})

    ctrl.sweep_export()
    assert state.sweep_error == ""
    out = Path(state.sweep_out_dir)
    assert out == tmp_path / "base-sweep"
    for rel in ("sweep.csv", "params.yaml", "Snakefile"):
        assert (out / rel).is_file(), f"missing {rel}"
    assert (out / "configs" / "base" / "setup.json").is_file()
    assert (out / "configs" / "variant-1" / "setup.json").is_file()
    # The edited variant (not the base) is what got materialized.
    variant_setup = json.loads(
        (out / "configs" / "variant-1" / "setup.json").read_text()
    )
    assert variant_setup["transport_properties_config"]["nu"] == 2e-5
    assert state.sweep_case_count == 2
    assert "snakemake" in state.sweep_status


def test_load_exported_rebuilds_canvas(tmp_path):
    # V4: export a sweep, then load it back onto a fresh canvas.
    server = build_app(server=get_server("neofoam_ui_test_sweep_export_rt"))
    state, ctrl = server.state, server.controller
    _seed_defaults(server)
    state.target_dir = str(tmp_path / "base")
    ctrl.save_case()

    ctrl.sweep_toggle_dimension("transport_properties_config")
    node_id = "dim:transport_properties_config"
    ctrl.sweep_variant_add(node_id)
    ctrl.sweep_rename_buffer(node_id, "nu2")
    ctrl.sweep_variant_rename(node_id)
    ctrl.sweep_variant_edit(node_id, {"transportModel": "Newtonian", "nu": 2e-5})
    ctrl.sweep_export()
    out_dir = state.sweep_out_dir
    exported_variants = dict(_dim_nodes(server)[0]["data"]["entries"])

    # Fresh server: load the exported sweep from the directory field.
    server2 = build_app(server=get_server("neofoam_ui_test_sweep_load_rt"))
    s2, c2 = server2.state, server2.controller
    assert _dim_nodes(server2) == []
    s2.sweep_out_dir = out_dir
    c2.sweep_load()

    (node,) = _dim_nodes(server2)
    assert node["id"] == "dim:transport_properties_config"
    assert node["data"]["entries"] == exported_variants
    # The default pipeline rules and the Configure mirror are restored.
    assert s2.sweep_dims_on_canvas == ["transport_properties_config"]
    assert s2.sweep_cfg_dim == "transport_properties_config"
    assert set(s2.sweep_cfg_variants) == {"base", "nu2"}
    assert s2.sweep_case_count == 2
    # A freshly loaded sweep matches disk — not dirty.
    assert not s2.sweep_dirty
    assert "Loaded 1 dimension" in s2.sweep_status


def test_load_exported_restores_cad_axis(tmp_path):
    # V4 + CAD: a round-tripped CAD sweep reappears on the canvas (the axis is
    # otherwise silently dropped since it has no solver-config entry).
    server = build_app(server=get_server("neofoam_ui_test_sweep_cad_rt_export"))
    state, ctrl = server.state, server.controller
    _seed_defaults(server)
    state.target_dir = str(tmp_path / "base")
    ctrl.save_case()

    ctrl.sweep_toggle_dimension("transport_properties_config")
    ctrl.sweep_add_cad_dimension("geometry/design.FCStd", {"R_mm": 8.0})
    dim = "dim:cad"
    ctrl.sweep_rename_buffer(dim, "R8")
    ctrl.sweep_variant_rename(dim)
    ctrl.sweep_variant_add(dim)
    ctrl.sweep_rename_buffer(dim, "R10")
    ctrl.sweep_variant_rename(dim)
    ctrl.sweep_variant_edit(dim, {"R_mm": 10.0})
    ctrl.sweep_export()
    assert state.sweep_error == ""
    out_dir = state.sweep_out_dir

    # Fresh server: load the exported cad sweep from the directory field.
    server2 = build_app(server=get_server("neofoam_ui_test_sweep_cad_rt_load"))
    s2, c2 = server2.state, server2.controller
    s2.sweep_out_dir = out_dir
    c2.sweep_load()

    assert "cad" in s2.sweep_dims_on_canvas
    cad_node = next(n for n in _dim_nodes(server2) if n["id"] == "dim:cad")
    assert set(cad_node["data"]["entries"]) == {"R8", "R10"}
    assert cad_node["data"]["entries"]["R10"] == {"R_mm": 10.0}


def test_load_exported_reports_missing_directory():
    server = build_app(server=get_server("neofoam_ui_test_sweep_load_miss"))
    state, ctrl = server.state, server.controller
    state.sweep_out_dir = "/nonexistent/sweep-dir"
    ctrl.sweep_load()
    assert "Could not load sweep" in state.sweep_error
    assert _dim_nodes(server) == []


def test_mesh_source_palette_toggles_keyed_mesh_dimension():
    # blockMesh/snappy are offered as mesh sources (not physics-config dimensions)
    # and toggle the single reserved keyed ``mesh`` dimension on the canvas.
    server = build_app(server=get_server("neofoam_ui_test_sweep_mesh_palette"))
    state, ctrl = server.state, server.controller

    values = {item["value"] for item in state.sweep_mesh_palette}
    assert {"block_mesh_dict_config", "snappy_hex_mesh_dict_config"} <= values
    # The mesh dicts must NOT leak into the physics-config palette.
    cfg_values = {item["value"] for item in state.sweep_config_palette}
    assert "block_mesh_dict_config" not in cfg_values

    ctrl.sweep_add_mesh_source("block_mesh_dict_config")
    assert state.sweep_mesh_on_canvas == ["block_mesh_dict_config"]
    (mesh_node,) = [n for n in _dim_nodes(server) if n["data"]["dim"] == "mesh"]
    assert mesh_node["id"] == "dim:mesh"
    # A second source joins the same mesh dimension (config-name-keyed payload).
    ctrl.sweep_add_mesh_source("snappy_hex_mesh_dict_config")
    assert state.sweep_mesh_on_canvas == [
        "block_mesh_dict_config",
        "snappy_hex_mesh_dict_config",
    ]
    assert len([n for n in _dim_nodes(server) if n["data"]["dim"] == "mesh"]) == 1

    # Toggling one off keeps the dimension; toggling the last off removes it.
    ctrl.sweep_remove_mesh_source("block_mesh_dict_config")
    assert state.sweep_mesh_on_canvas == ["snappy_hex_mesh_dict_config"]
    ctrl.sweep_remove_mesh_source("snappy_hex_mesh_dict_config")
    assert state.sweep_mesh_on_canvas == []
    assert not any(n["data"]["dim"] == "mesh" for n in _dim_nodes(server))


def test_mesh_dimension_exports_config_name_keyed(tmp_path):
    # A swept mesh dimension exports as the reserved keyed ``mesh`` axis whose
    # variant payloads are config-name-keyed (the shape the runner applies in the
    # meshes/{variant} mini-case).
    from neofoam.workflow.paramspace import read_params_yaml

    server = build_app(server=get_server("neofoam_ui_test_sweep_mesh_export"))
    state, ctrl = server.state, server.controller
    _seed_defaults(server)
    state.target_dir = str(tmp_path / "base")
    ctrl.save_case()

    ctrl.sweep_add_mesh_source("block_mesh_dict_config")
    assert state.sweep_valid
    ctrl.sweep_export()
    assert not state.sweep_error
    assert state.sweep_exported

    params = read_params_yaml(Path(state.sweep_out_dir) / "params.yaml")
    assert "mesh" in params
    (variant,) = params["mesh"].values()
    assert "block_mesh_dict_config" in variant

    # The workflow round-trips: reopening restores the mesh dimension.
    server2 = build_app(server=get_server("neofoam_ui_test_sweep_mesh_reload"))
    server2.state.sweep_out_dir = state.sweep_out_dir
    server2.controller.sweep_load()
    assert server2.state.sweep_mesh_on_canvas == ["block_mesh_dict_config"]


def test_field_source_adds_per_case_dimension_and_exports(tmp_path):
    # A whole 0/<field> config (here 0/U) is sweepable as a regular per-case
    # dimension — e.g. to vary the inlet velocity.
    server = build_app(server=get_server("neofoam_ui_test_sweep_field"))
    state, ctrl = server.state, server.controller

    values = {item["value"] for item in state.sweep_field_palette}
    assert "u_field_config" in values
    # Field configs are NOT in the physics-config palette (kept as split halves).
    assert "u_field_config" not in {i["value"] for i in state.sweep_config_palette}

    _seed_defaults(server)
    state.target_dir = str(tmp_path / "base")
    ctrl.save_case()

    ctrl.sweep_add_field_source("u_field_config")
    assert state.sweep_field_on_canvas == ["u_field_config"]
    # It is a normal (case) dimension, not the keyed mesh axis.
    (node,) = [n for n in _dim_nodes(server) if n["data"]["dim"] == "u_field_config"]
    assert node["id"] == "dim:u_field_config"

    # Two inlet-speed variants → the config-payload boundaryField.inlet.value.
    ndim = "dim:u_field_config"
    base_u = dict(state.sweep_cfg_data)

    def _with_inlet(speed):
        bf = {**base_u.get("boundaryField", {})}
        bf["inlet"] = {"type": "fixedValue", "value": f"uniform ({speed} 0 0)"}
        return {**base_u, "boundaryField": bf}

    ctrl.sweep_rename_buffer(ndim, "u1")
    ctrl.sweep_variant_rename(ndim)
    ctrl.sweep_variant_edit(ndim, _with_inlet(1))
    ctrl.sweep_variant_add(ndim)
    ctrl.sweep_rename_buffer(ndim, "u2")
    ctrl.sweep_variant_rename(ndim)
    ctrl.sweep_variant_edit(ndim, _with_inlet(2))

    assert state.sweep_valid
    ctrl.sweep_export()
    assert not state.sweep_error

    from neofoam.workflow.paramspace import read_params_yaml

    params = read_params_yaml(Path(state.sweep_out_dir) / "params.yaml")
    assert set(params["u_field_config"]) == {"u1", "u2"}
    assert (
        params["u_field_config"]["u2"]["boundaryField"]["inlet"]["value"]
        == "uniform (2 0 0)"
    )


def _export_transport_sweep(server, tmp_path) -> str:
    """Save + export a single-dimension sweep; return its output directory."""
    state, ctrl = server.state, server.controller
    _seed_defaults(server)
    state.target_dir = str(tmp_path / "base")
    ctrl.save_case()
    ctrl.sweep_toggle_dimension("transport_properties_config")
    ctrl.sweep_export()
    return str(state.sweep_out_dir)


def test_load_exported_skips_unsupported_dimension(tmp_path):
    # A dimension with no matching config on this solver (an unknown config name)
    # is reported and skipped; the supported dimension still loads.
    from neofoam.workflow.paramspace import read_params_yaml, write_params_yaml

    server = build_app(server=get_server("neofoam_ui_test_sweep_skip_export"))
    out_dir = _export_transport_sweep(server, tmp_path)

    params_path = Path(out_dir) / "params.yaml"
    params = read_params_yaml(params_path)
    params["nonexistent_config"] = {"coarse": {"cells": 1}}
    write_params_yaml(params_path, params)

    server2 = build_app(server=get_server("neofoam_ui_test_sweep_skip_load"))
    s2, c2 = server2.state, server2.controller
    s2.sweep_out_dir = out_dir
    c2.sweep_load()

    assert s2.sweep_dims_on_canvas == ["transport_properties_config"]
    assert "skipped unsupported dimension" in s2.sweep_status
    assert s2.sweep_cfg_dim == "transport_properties_config"


def test_load_exported_all_unsupported_clears_tab(tmp_path):
    # An all-unsupported load leaves an empty canvas and a cleared Configure
    # mirror without crashing.
    from neofoam.workflow.paramspace import write_params_yaml

    server = build_app(server=get_server("neofoam_ui_test_sweep_allunsup_export"))
    out_dir = _export_transport_sweep(server, tmp_path)

    write_params_yaml(
        Path(out_dir) / "params.yaml", {"nonexistent_config": {"coarse": {"cells": 1}}}
    )

    server2 = build_app(server=get_server("neofoam_ui_test_sweep_allunsup_load"))
    s2, c2 = server2.state, server2.controller
    s2.sweep_out_dir = out_dir
    c2.sweep_load()

    assert s2.sweep_cfg_dim == ""
    assert s2.sweep_dims_on_canvas == []
    assert s2.sweep_cfg_data == {}


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="snakemake not installed")
def test_refresh_dag_renders_snakemake_graph(tmp_path):
    server = build_app(server=get_server("neofoam_ui_test_sweep_dag"))
    state, ctrl = server.state, server.controller
    _seed_defaults(server)
    state.target_dir = str(tmp_path / "base")
    ctrl.save_case()
    assert state.scaffolded

    ctrl.sweep_toggle_dimension("transport_properties_config")
    ctrl.sweep_refresh_dag()

    assert state.sweep_dag_error == ""
    dag_nodes = ctrl.sweep_get_dag_nodes()
    labels = {n["data"]["label"].split("\n")[0] for n in dag_nodes}
    # One job per pipeline rule (single case, single implicit mesh variant).
    assert {"all", "setup", "solve", "setup_mesh", "blockMesh"} <= labels
