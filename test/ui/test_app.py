# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless build + nav + save round-trip for the trame app (no browser)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")
pytest.importorskip("trame")
pytest.importorskip("trame_flow")  # build_app renders the sweep canvas

from trame.app import get_server  # noqa: E402

from neofoam.mcp import tools  # noqa: E402
from neofoam.mcp.registry import resolve_solver  # noqa: E402
from neofoam.ui import build_app  # noqa: E402
from neofoam.ui.app import _schema_key  # noqa: E402
from neofoam.ui.case_load import apply_configs_to_forms, read_case_configs  # noqa: E402
from neofoam.ui.geometry import discover_geometry  # noqa: E402
from neofoam.ui.steps import build_model_families  # noqa: E402

#: How long the stubbed STL read blocks — a 21 MB STL takes ~0.5 s in practice.
_SCAN_SECONDS = 0.3

#: A checked-in ``simulationType laminar`` case (read-only).
_LAMINAR_CASE = Path(__file__).resolve().parents[1] / "setup_pimple"

#: The checked-in STL folder the geometry-handler tests scan (read-only).
_TRI_SURFACE = (
    Path(__file__).resolve().parents[1]
    / "tooling"
    / "workflow"
    / "cases"
    / "tube_bank"
    / "constant"
    / "triSurface"
)


def test_build_app_constructs():
    # plugins=[] pins the built-in baseline regardless of any installed
    # `neofoam.ui.steps` entry points (e.g. the optional FoamCAD step).
    server = build_app(server=get_server("neofoam_ui_test_construct"), plugins=[])
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
        "sweep",
        "review",
    ]
    for entry in entries:
        assert server.state[entry.state_key] is not None
        assert server.state[_schema_key(entry)] == entry.schema


def test_nav_does_not_reset_form_state():
    server = build_app(server=get_server("neofoam_ui_test_nav"), plugins=[])
    entries = server.controller.get_entries()
    before = {e.state_key for e in entries}

    server.state.current_step = "bcs"  # navigate
    after = {e.state_key for e in server.controller.get_entries()}
    assert before == after  # same entries/state keys — nothing rebuilt


def test_save_case_round_trip(tmp_path):
    solver = resolve_solver("incompressibleFluid")
    server = build_app(server=get_server("neofoam_ui_test_save"), plugins=[])
    entries = server.controller.get_entries()

    defaults = tools.config_schema(solver, "transport_properties_config").defaults
    for entry in entries:
        server.state[entry.state_key] = (
            {**defaults, "nu": 1e-05} if entry.config_name == "transport_properties_config" else {}
        )

    server.state.target_dir = str(tmp_path)
    server.controller.save_case()

    assert (tmp_path / "constant" / "transportProperties").is_file()
    assert server.state.save_report is not None
    assert any("transportProperties" in w for w in server.state.save_report["written"])


def test_save_scaffolds_and_validates(tmp_path):
    import os  # noqa: PLC0415

    solver = resolve_solver("incompressibleFluid")
    server = build_app(server=get_server("neofoam_ui_test_scaffold"), plugins=[])
    entries = server.controller.get_entries()

    defaults = tools.config_schema(solver, "transport_properties_config").defaults
    for entry in entries:
        server.state[entry.state_key] = (
            {**defaults, "nu": 1e-05} if entry.config_name == "transport_properties_config" else {}
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
    import shutil  # noqa: PLC0415

    # A case whose constant/triSurface holds the given STLs (copied from tube_bank).
    src_tri = (
        Path(__file__).resolve().parents[1]
        / "tooling"
        / "workflow"
        / "cases"
        / "tube_bank"
        / "constant"
        / "triSurface"
    )
    dst_tri = tmp_path / "constant" / "triSurface"
    shutil.copytree(src_tri, dst_tri)

    server = build_app(server=get_server("neofoam_ui_test_geometry"), plugins=[])
    # Point the STL-folder field straight at the triSurface dir (as in the UI).
    server.state.stl_dir = str(dst_tri)
    server.state.target_dir = str(tmp_path)

    asyncio.run(server.controller.load_geometry())
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
    server = build_app(server=get_server("neofoam_ui_test_badsave"), plugins=[])
    server.state.target_dir = str(tmp_path)
    server.controller.save_case()  # must not raise

    assert server.state.current_step == "review"
    assert "error" in server.state.save_report
    assert server.state.validation_ok is False
    assert any(f["level"] == "error" for f in server.state.findings)


def _seed_transport_defaults(server) -> None:
    """The minimal valid form state (only transportProperties filled)."""
    solver = resolve_solver("incompressibleFluid")
    defaults = tools.config_schema(solver, "transport_properties_config").defaults
    for entry in server.controller.get_entries():
        server.state[entry.state_key] = (
            {**defaults, "nu": 1e-05} if entry.config_name == "transport_properties_config" else {}
        )


def test_save_case_with_a_blank_target_writes_nothing(tmp_path, monkeypatch):
    # A blank target field used to resolve to the server's launch directory, so a
    # filled wizard wrote a whole case into the user's checkout.
    server = build_app(server=get_server("neofoam_ui_test_blank_target"), plugins=[])
    _seed_transport_defaults(server)
    monkeypatch.chdir(tmp_path)

    server.state.target_dir = ""
    server.controller.save_case()

    assert list(tmp_path.iterdir()) == []
    assert "error" in server.state.save_report
    assert server.state.validation_ok is False
    assert any("target directory" in f["message"] for f in server.state.findings)


def test_revalidate_with_a_blank_target_reports_instead_of_validating_the_launch_dir(
    tmp_path, monkeypatch
):
    # Path("").is_dir() is True, so an unguarded revalidate reported findings about
    # whatever directory the server was started from.
    server = build_app(server=get_server("neofoam_ui_test_blank_reval"), plugins=[])
    monkeypatch.chdir(tmp_path)

    server.state.target_dir = ""
    server.controller.revalidate()

    assert server.state.validation_ok is False
    assert [f["message"] for f in server.state.findings] == [
        "No target directory — type an absolute path first."
    ]


def test_save_case_reports_a_scaffold_failure(tmp_path):
    # A case root the scaffold cannot write into (here: an `Allrun` directory in the
    # way, standing in for a read-only root). The configs are written, but the wizard
    # must not land on Review still telling the user to click Save.
    (tmp_path / "Allrun").mkdir()
    server = build_app(server=get_server("neofoam_ui_test_scaffold_fail"), plugins=[])
    _seed_transport_defaults(server)
    server.state.target_dir = str(tmp_path)

    server.controller.save_case()  # must not raise

    assert server.state.current_step == "review"
    assert server.state.validation_ok is False  # not None → Review shows the failure
    assert server.state.scaffolded == []
    assert "error" in server.state.save_report
    assert server.state.findings


def test_failed_scan_clears_the_mesh_written_alert(tmp_path):
    import shutil  # noqa: PLC0415

    src_tri = (
        Path(__file__).resolve().parents[1]
        / "tooling"
        / "workflow"
        / "cases"
        / "tube_bank"
        / "constant"
        / "triSurface"
    )
    shutil.copytree(src_tri, tmp_path / "constant" / "triSurface")

    server = build_app(server=get_server("neofoam_ui_test_scan_fail"), plugins=[])
    server.state.stl_dir = str(tmp_path / "constant" / "triSurface")
    server.state.target_dir = str(tmp_path)
    asyncio.run(server.controller.load_geometry())
    server.controller.write_mesh()
    assert server.state.mesh_written  # the green "Wrote: …" alert is up

    server.state.stl_dir = str(tmp_path / "does_not_exist")
    asyncio.run(server.controller.load_geometry())

    # The failure must not sit above a stale success alert, and must not read as info.
    assert server.state.geometry_patches == []
    assert server.state.mesh_written == []
    assert server.state.geometry_severity == "error"
    assert "Could not read geometry" in server.state.geometry_status


def _slow_scan(spec, scanned: list[str], seconds: float = _SCAN_SECONDS):
    """Stand-in for reading a large STL: blocks for ``seconds``, then yields ``spec``."""

    def scan(path, **_kwargs):
        scanned.append(str(path))
        time.sleep(seconds)
        return spec

    return scan


def test_scan_keeps_the_event_loop_running(monkeypatch, heartbeat_ticks):
    # trame is single-threaded, so an STL read on the loop freezes the whole UI —
    # every other client callback, including the heartbeat, stops for its duration.
    server = build_app(server=get_server("neofoam_ui_test_scan_loop"), plugins=[])
    server.state.stl_dir = str(_TRI_SURFACE)
    monkeypatch.setattr(
        "neofoam.ui.app.discover_geometry", _slow_scan(discover_geometry(_TRI_SURFACE), [])
    )

    ticks = heartbeat_ticks(server.controller.load_geometry)

    assert ticks >= 5  # ~15 over a 0.3 s scan; 0 while the loop is blocked
    assert server.state.geometry_patches  # and the scan still landed


def test_scan_is_busy_while_it_runs(monkeypatch):
    # Without a busy flag the Scan button looks idle through the whole freeze.
    server = build_app(server=get_server("neofoam_ui_test_scan_busy"), plugins=[])
    server.state.stl_dir = str(_TRI_SURFACE)
    spec = discover_geometry(_TRI_SURFACE)
    busy_while_scanning: list[bool] = []
    monkeypatch.setattr(
        "neofoam.ui.app.discover_geometry",
        lambda *_a, **_kw: (busy_while_scanning.append(server.state.geometry_busy), spec)[1],
    )

    asyncio.run(server.controller.load_geometry())

    assert busy_while_scanning == [True]
    assert server.state.geometry_busy is False


def test_scan_started_while_one_runs_is_dropped(monkeypatch):
    # Clicks queued during the freeze all land once it ends; a second scan would
    # re-seed the boundary-condition forms underneath the first one's results.
    server = build_app(server=get_server("neofoam_ui_test_scan_reentry"), plugins=[])
    server.state.stl_dir = str(_TRI_SURFACE)
    scanned: list[str] = []
    monkeypatch.setattr(
        "neofoam.ui.app.discover_geometry", _slow_scan(discover_geometry(_TRI_SURFACE), scanned)
    )

    async def drive() -> None:
        first = asyncio.create_task(server.controller.load_geometry())
        await asyncio.sleep(_SCAN_SECONDS / 3)  # the first scan is in flight
        await server.controller.load_geometry()  # a click queued during it
        await first

    asyncio.run(drive())

    assert scanned == [str(_TRI_SURFACE)]
    assert server.state.geometry_busy is False


def test_revalidate_reruns_without_resaving(tmp_path):
    solver = resolve_solver("incompressibleFluid")
    server = build_app(server=get_server("neofoam_ui_test_reval"), plugins=[])
    entries = server.controller.get_entries()
    defaults = tools.config_schema(solver, "transport_properties_config").defaults
    for entry in entries:
        server.state[entry.state_key] = (
            {**defaults, "nu": 1e-05} if entry.config_name == "transport_properties_config" else {}
        )
    server.state.target_dir = str(tmp_path)
    server.controller.save_case()

    allrun_mtime = (tmp_path / "Allrun").stat().st_mtime
    server.controller.revalidate()  # must not raise, must not re-scaffold
    assert (tmp_path / "Allrun").stat().st_mtime == allrun_mtime
    assert server.state.validation_ok is not None


def test_pick_one_family_starts_on_one_member_and_switching_deselects_the_other():
    server = build_app(server=get_server("neofoam_ui_test_family"), plugins=[])
    state, ctrl = server.state, server.controller

    # Pimple and Simple want contradictory ddtSchemes — exactly one is ever selected.
    assert state.choice_PressureVelocityAlgorithm == "Pimple"
    assert state.sel_Pimple is True
    assert state.sel_Simple is False

    ctrl.select_model("Simple")

    assert state.choice_PressureVelocityAlgorithm == "Simple"
    assert state.sel_Pimple is False
    assert state.sel_Simple is True


def test_save_case_writes_only_the_chosen_algorithm(tmp_path):
    server = build_app(server=get_server("neofoam_ui_test_family_save"), plugins=[])
    _seed_transport_defaults(server)

    server.state.target_dir = str(tmp_path / "pimple")
    server.controller.save_case()
    pimple = (tmp_path / "pimple" / "system" / "fvSolution").read_text()
    assert "PIMPLE" in pimple
    assert "SIMPLE" not in pimple  # the unselected algorithm's block is not written

    server.controller.select_model("Simple")
    server.state.target_dir = str(tmp_path / "simple")
    server.controller.save_case()
    simple = (tmp_path / "simple" / "system" / "fvSolution").read_text()
    assert "SIMPLE" in simple
    assert "PIMPLE" not in simple


def test_save_case_writes_no_piso_block_beside_pimple(tmp_path):
    # A PISO block is only read when no PIMPLE block exists, so writing both left
    # the user a block (and two wizard panels) whose edits never took effect.
    server = build_app(server=get_server("neofoam_ui_test_no_piso"), plugins=[])
    _seed_transport_defaults(server)

    server.state.target_dir = str(tmp_path)
    server.controller.save_case()

    fv_solution = (tmp_path / "system" / "fvSolution").read_text()
    assert "PIMPLE" in fv_solution
    assert "PISO" not in fv_solution


def test_save_case_writes_a_hand_added_scheme_entry_as_openfoam_tokens(tmp_path):
    # The Numerics step's "+ add entry" puts a key the schema does not declare into a
    # scheme section, as the same structured object its declared siblings hold.
    server = build_app(server=get_server("neofoam_ui_test_added_scheme"), plugins=[])
    schemes = server.state.form_pimple_fv_schemes
    _seed_transport_defaults(server)
    server.state.form_pimple_fv_schemes = {
        **schemes,
        "divSchemes": {
            **schemes["divSchemes"],
            "div(phi,k)": {
                "type": "Gauss",
                "interpolation": {"type": "linearUpwind", "grad_field": "grad(k)"},
            },
            "div(phi,epsilon)": {"type": "none"},  # added, scheme not picked yet
        },
    }

    server.state.target_dir = str(tmp_path)
    server.controller.save_case()

    written = (tmp_path / "system" / "fvSchemes").read_text().splitlines()
    entries = [" ".join(line.split()) for line in written]
    assert "div(phi,k) Gauss linearUpwind grad(k);" in entries
    assert "div(phi,epsilon) none;" in entries


def _turbulence_properties(server) -> dict:
    entries = {e.cls_name: e for e in server.controller.get_entries()}
    return server.state[entries["TurbulencePropertiesConfig"].state_key]


@pytest.mark.parametrize(
    ("solver_name", "expected"),
    [
        (
            "incompressibleFluid",
            {
                "simulationType": "RAS",
                "RAS": {"RASModel": "kEpsilon", "turbulence": True, "printCoeffs": False},
            },
        ),
        (
            "incompressibleFluidNeoN",
            {
                "simulationType": "RAS",
                "RAS": {"RASModel": "kEpsilon", "turbulence": True, "printCoeffs": False},
            },
        ),
        # No momentum-transport family to choose from: the two-phase backend reads the
        # file itself, and laminar is the only state needing no further input.
        ("incompressibleVoF", {"simulationType": "laminar"}),
    ],
)
def test_turbulence_properties_start_on_the_selected_model(solver_name, expected):
    server = build_app(
        solver_name=solver_name,
        server=get_server(f"neofoam_ui_test_turbulence_default_{solver_name}"),
        plugins=[],
    )

    assert _turbulence_properties(server) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("laminar", {"simulationType": "laminar"}),
        (
            "kOmegaSST",
            {
                "simulationType": "RAS",
                "RAS": {"RASModel": "kOmegaSST", "turbulence": True, "printCoeffs": False},
            },
        ),
    ],
)
def test_selecting_a_turbulence_model_rewrites_turbulence_properties(model, expected):
    server = build_app(server=get_server(f"neofoam_ui_test_turbulence_{model}"), plugins=[])

    server.controller.select_model(model)

    assert _turbulence_properties(server) == expected


def test_selecting_another_family_leaves_turbulence_properties_alone():
    server = build_app(server=get_server("neofoam_ui_test_turbulence_untouched"), plugins=[])
    server.controller.select_model("laminar")

    server.controller.select_model("Simple")

    assert _turbulence_properties(server) == {"simulationType": "laminar"}


def test_loaded_case_overrides_the_default_turbulence_properties():
    solver = resolve_solver("incompressibleFluid")
    server = build_app(server=get_server("neofoam_ui_test_turbulence_loaded"), plugins=[])
    configs = read_case_configs(_LAMINAR_CASE, solver)

    apply_configs_to_forms(
        server.state, server.controller.get_entries(), build_model_families(solver), configs
    )

    assert _turbulence_properties(server) == {"simulationType": "laminar"}


def test_loaded_case_moves_the_turbulence_choice_to_the_loaded_model():
    # turbulenceProperties is owned by no single model, so filling it selects none:
    # the radio group has to be moved to the model the loaded file names.
    solver = resolve_solver("incompressibleFluid")
    server = build_app(server=get_server("neofoam_ui_test_turbulence_loaded_choice"), plugins=[])
    configs = read_case_configs(_LAMINAR_CASE, solver)

    apply_configs_to_forms(
        server.state, server.controller.get_entries(), build_model_families(solver), configs
    )

    assert server.state.choice_momentumTransportModel == "laminar"
    assert server.state.sel_laminar is True
    assert server.state.sel_kEpsilon is False


def test_panel_chip_shows_the_owning_models_label():
    server = build_app(server=get_server("neofoam_ui_test_chip_label"), plugins=[])

    template = server.state["trame__template_main"]
    assert "Adaptive time step (Courant)\n</VChip>" in template
    assert "\ncourant\n</VChip>" not in template


def test_save_case_writes_the_selected_turbulence_model(tmp_path):
    server = build_app(server=get_server("neofoam_ui_test_turbulence_save"), plugins=[])
    _seed_transport_defaults(server)
    server.controller.select_model("kOmegaSST")
    server.state.target_dir = str(tmp_path)

    server.controller.save_case()

    written = (tmp_path / "constant" / "turbulenceProperties").read_text()
    assert "kOmegaSST" in written


@pytest.mark.parametrize(
    ("solver_name", "shown"),
    [
        ("incompressibleFluid", True),  # Newtonian is always on
        ("incompressibleFluidNeoN", False),  # every required model is a family choice
    ],
)
def test_included_models_heading_needs_an_always_on_model(solver_name, shown):
    server = build_app(
        solver_name=solver_name,
        server=get_server(f"neofoam_ui_test_included_{solver_name}"),
        plugins=[],
    )

    assert ("Included models" in server.state["trame__template_main"]) is shown


def test_boundary_conditions_step_explains_its_empty_state():
    # Before a scan each field panel is a bare "Property Name [+]" row; the step says
    # where patches come from and that the row adds one by hand, until a scan ran.
    server = build_app(server=get_server("neofoam_ui_test_bcs_hint"), plugins=[])

    template = server.state["trame__template_main"]
    assert "run Scan in the Geometry step to seed them" in template
    assert 'v-show="!geometry_patches.length"' in template
    # The adder row is labelled "Patch name" itself, so the hint need not name the box.
    assert "Property Name" not in template


def test_forms_receive_the_adder_translations():
    from neofoam.ui.forms import ADDER_TRANSLATIONS  # noqa: PLC0415

    server = build_app(server=get_server("neofoam_ui_test_translations"), plugins=[])

    assert ':translations="form_translations"' in server.state["trame__template_main"]
    assert server.state["form_translations"] == ADDER_TRANSLATIONS
