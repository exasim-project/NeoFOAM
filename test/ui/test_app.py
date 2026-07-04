# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless build + nav + save round-trip for the trame app (no browser)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pybFoam")
pytest.importorskip("trame")

from trame.app import get_server  # noqa: E402

from neofoam.mcp import tools  # noqa: E402
from neofoam.mcp.registry import resolve_solver  # noqa: E402
from neofoam.ui import build_app  # noqa: E402
from neofoam.ui.app import _schema_key  # noqa: E402


def test_build_app_constructs():
    server = build_app(server=get_server("neofoam_ui_test_construct"))
    assert callable(server.controller.save_case)
    assert server.state.current_step == "models"
    assert server.state.target_dir == ""
    entries = server.controller.get_entries()
    steps = server.controller.get_steps()
    assert len(entries) > 0
    assert [s.id for s in steps] == [
        "models",
        "geometry",
        "bcs",
        "initial",
        "schemes",
        "review",
    ]
    for entry in entries:
        assert server.state[entry.state_key] is not None
        assert server.state[_schema_key(entry)] == entry.schema


def test_nav_does_not_reset_form_state():
    server = build_app(server=get_server("neofoam_ui_test_nav"))
    entries = server.controller.get_entries()
    before = {e.state_key for e in entries}

    server.state.current_step = "bcs"  # navigate
    after = {e.state_key for e in server.controller.get_entries()}
    assert before == after  # same entries/state keys — nothing rebuilt


def test_save_case_round_trip(tmp_path):
    solver = resolve_solver("incompressibleFluid")
    server = build_app(server=get_server("neofoam_ui_test_save"))
    entries = server.controller.get_entries()

    defaults = tools.config_schema(solver, "transport_properties_config").defaults
    for entry in entries:
        server.state[entry.state_key] = (
            dict(defaults) if entry.config_name == "transport_properties_config" else {}
        )

    server.state.target_dir = str(tmp_path)
    server.controller.save_case()

    assert (tmp_path / "constant" / "transportProperties").is_file()
    assert server.state.save_report is not None
    assert any("transportProperties" in w for w in server.state.save_report["written"])


def test_save_scaffolds_and_validates(tmp_path):
    import os

    solver = resolve_solver("incompressibleFluid")
    server = build_app(server=get_server("neofoam_ui_test_scaffold"))
    entries = server.controller.get_entries()

    defaults = tools.config_schema(solver, "transport_properties_config").defaults
    for entry in entries:
        server.state[entry.state_key] = (
            dict(defaults) if entry.config_name == "transport_properties_config" else {}
        )
    server.state.target_dir = str(tmp_path)
    server.controller.save_case()

    # Runnable-case scaffold written + executable.
    allrun, allclean = tmp_path / "Allrun", tmp_path / "Allclean"
    assert allrun.is_file() and os.access(allrun, os.X_OK)
    assert allclean.is_file() and os.access(allclean, os.X_OK)
    assert server.state.scaffolded

    # Validation ran and the wizard advanced to Review.
    assert server.state.validation_ok is not None
    assert isinstance(server.state.findings, list)
    assert server.state.current_step == "review"


def test_geometry_scan_and_write_mesh(tmp_path):
    import shutil

    # A case whose constant/triSurface holds the given STLs (copied from tube_bank).
    src_tri = (
        Path(__file__).resolve().parents[1]
        / "e2e"
        / "cases"
        / "tube_bank"
        / "constant"
        / "triSurface"
    )
    dst_tri = tmp_path / "constant" / "triSurface"
    shutil.copytree(src_tri, dst_tri)

    server = build_app(server=get_server("neofoam_ui_test_geometry"))
    # Point the STL-folder field straight at the triSurface dir (as in the UI).
    server.state.stl_dir = str(dst_tri)
    server.state.target_dir = str(tmp_path)

    server.controller.load_geometry()
    names = {p["name"] for p in server.state.geometry_patches}
    assert names == {"inlet", "outlet", "walls", "frontBack", "tubes"}
    assert server.state.geo_bbox is not None

    server.controller.write_mesh()
    assert (tmp_path / "system" / "blockMeshDict").is_file()
    assert (tmp_path / "system" / "snappyHexMeshDict").is_file()
    assert (tmp_path / "system" / "preprocess.yaml").is_file()
    assert server.state.mesh_written


def test_incomplete_save_is_reported_not_raised(tmp_path):
    # The pristine app pre-seeds partial defaults (e.g. controlDict lacks endTime);
    # saving must surface the error in Review, not crash the controller.
    server = build_app(server=get_server("neofoam_ui_test_badsave"))
    server.state.target_dir = str(tmp_path)
    server.controller.save_case()  # must not raise

    assert server.state.current_step == "review"
    assert "error" in server.state.save_report
    assert server.state.validation_ok is False
    assert any(f["level"] == "error" for f in server.state.findings)


def test_revalidate_reruns_without_resaving(tmp_path):
    solver = resolve_solver("incompressibleFluid")
    server = build_app(server=get_server("neofoam_ui_test_reval"))
    entries = server.controller.get_entries()
    defaults = tools.config_schema(solver, "transport_properties_config").defaults
    for entry in entries:
        server.state[entry.state_key] = (
            dict(defaults) if entry.config_name == "transport_properties_config" else {}
        )
    server.state.target_dir = str(tmp_path)
    server.controller.save_case()

    allrun_mtime = (tmp_path / "Allrun").stat().st_mtime
    server.controller.revalidate()  # must not raise, must not re-scaffold
    assert (tmp_path / "Allrun").stat().st_mtime == allrun_mtime
    assert server.state.validation_ok is not None
