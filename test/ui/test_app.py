# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless build + nav + save round-trip for the trame app (no browser)."""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")
pytest.importorskip("trame")
pytest.importorskip("trame_flow")  # build_app renders the sweep canvas

from trame.app import get_server  # noqa: E402

from neofoam.ui import build_app  # noqa: E402
from neofoam.ui.forms import schema_key  # noqa: E402
from neofoam.ui.geometry import discover_geometry  # noqa: E402

#: How long the stubbed STL read blocks — a 21 MB STL takes ~0.5 s in practice.
_SCAN_SECONDS = 0.3

#: A real on-disk case the Load case button opens (read-only).
_PITZ_DAILY = (
    Path(__file__).resolve().parents[1] / "solver" / "incompressibleFluid" / "val_pitzDaily"
)

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


def test_build_app_constructs(wizard):
    # The `wizard` fixture's plugins=[] pins the built-in baseline regardless of any installed
    # `neofoam.ui.steps` entry points (e.g. the optional FoamCAD step).
    assert callable(wizard.controller.save_case)
    assert wizard.state.current_step == "models"
    assert wizard.state.target_dir == ""
    entries = wizard.controller.get_entries()
    steps = wizard.controller.get_steps()
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
        assert wizard.state[entry.state_key] is not None
        assert wizard.state[schema_key(entry)] == entry.schema


def test_nav_does_not_reset_form_state(wizard):
    entries = wizard.controller.get_entries()
    before = {e.state_key for e in entries}

    wizard.state.current_step = "bcs"  # navigate
    after = {e.state_key for e in wizard.controller.get_entries()}
    assert before == after  # same entries/state keys — nothing rebuilt


def test_save_case_round_trip(tmp_path, seed_transport_defaults, wizard):
    seed_transport_defaults(wizard)

    wizard.state.target_dir = str(tmp_path)
    wizard.controller.save_case()

    assert (tmp_path / "constant" / "transportProperties").is_file()
    assert wizard.state.save_report is not None
    assert any("transportProperties" in w for w in wizard.state.save_report["written"])


def test_save_scaffolds_and_validates(tmp_path, seed_transport_defaults, wizard):
    import os  # noqa: PLC0415

    seed_transport_defaults(wizard)
    wizard.state.target_dir = str(tmp_path)
    wizard.controller.save_case()

    # Runnable-case scaffold written + executable.
    allrun, allclean = tmp_path / "Allrun", tmp_path / "Allclean"
    assert allrun.is_file() and os.access(allrun, os.X_OK)
    assert allclean.is_file() and os.access(allclean, os.X_OK)
    assert wizard.state.scaffolded

    # Validation ran and the wizard advanced to Review.
    assert wizard.state.validation_ok is not None
    assert isinstance(wizard.state.findings, list)
    assert wizard.state.current_step == "review"


def test_save_case_scaffolds_an_allrun_for_the_wizards_solver(tmp_path, seed_transport_defaults):
    neon_wizard = build_app(
        solver_name="incompressibleFluidNeoN",
        server=get_server("neofoam_ui_test_allrun_solver"),
        plugins=[],
    )
    seed_transport_defaults(neon_wizard)
    neon_wizard.state.target_dir = str(tmp_path)

    neon_wizard.controller.save_case()

    assert 'exec neofoam solver incompressiblefluidneon "$@"' in (tmp_path / "Allrun").read_text()


def _scan_tube_bank(wizard, tmp_path, *, set_target: bool = True) -> None:
    """Scan a copy of the tube-bank STLs, pointing the STL field straight at them."""
    dst_tri = tmp_path / "constant" / "triSurface"
    shutil.copytree(_TRI_SURFACE, dst_tri)
    wizard.state.stl_dir = str(dst_tri)
    if set_target:
        wizard.state.target_dir = str(tmp_path)
    asyncio.run(wizard.controller.load_geometry())


def test_geometry_scan_lists_the_stl_patches(tmp_path, wizard):
    _scan_tube_bank(wizard, tmp_path)

    names = {p["name"] for p in wizard.state.geometry_patches}
    assert names == {"inlet", "outlet", "walls", "frontBack", "tubes"}
    assert wizard.state.geo_bbox is not None


def test_write_mesh_writes_the_mesh_dicts_of_a_scanned_geometry(tmp_path, wizard):
    _scan_tube_bank(wizard, tmp_path)

    wizard.controller.write_mesh()

    assert (tmp_path / "system" / "blockMeshDict").is_file()
    assert (tmp_path / "system" / "snappyHexMeshDict").is_file()
    assert (tmp_path / "system" / "preprocess.yaml").is_file()
    assert wizard.state.mesh_written


def test_scan_pins_only_the_scanned_patches(tmp_path, wizard):
    # A pinned patch has no delete button. One added by hand before a re-scan is not
    # part of the geometry, so it has to stay deletable.
    entry = next(e for e in wizard.controller.get_entries() if e.key == "field_bc:UFieldConfig")
    wizard.state[entry.state_key] = {"boundaryField": {"byHand": {"type": "noSlip"}}}

    _scan_tube_bank(wizard, tmp_path, set_target=False)

    boundary_field = wizard.state[schema_key(entry)]["properties"]["boundaryField"]
    assert set(boundary_field["properties"]) == {"inlet", "outlet", "walls", "frontBack", "tubes"}
    assert "byHand" in wizard.state[entry.state_key]["boundaryField"]


def test_incomplete_save_is_reported_not_raised(tmp_path, wizard):
    # The pristine app pre-seeds partial defaults (e.g. controlDict lacks endTime);
    # saving must surface the error in Review, not crash the controller.
    wizard.state.target_dir = str(tmp_path)
    wizard.controller.save_case()  # must not raise

    assert wizard.state.current_step == "review"
    assert "error" in wizard.state.save_report
    assert wizard.state.validation_ok is False
    assert any(f["level"] == "error" for f in wizard.state.findings)


def test_save_case_with_a_blank_target_writes_nothing(
    tmp_path,
    monkeypatch,
    seed_transport_defaults,
    wizard,
):
    # A blank target field used to resolve to the server's launch directory, so a
    # filled wizard wrote a whole case into the user's checkout.
    seed_transport_defaults(wizard)
    monkeypatch.chdir(tmp_path)

    wizard.state.target_dir = ""
    wizard.controller.save_case()

    assert list(tmp_path.iterdir()) == []
    assert "error" in wizard.state.save_report
    assert wizard.state.validation_ok is False
    assert any("target directory" in f["message"] for f in wizard.state.findings)


def test_revalidate_with_a_blank_target_reports_instead_of_validating_the_launch_dir(
    tmp_path,
    monkeypatch,
    wizard,
):
    # Path("").is_dir() is True, so an unguarded revalidate reported findings about
    # whatever directory the server was started from.
    monkeypatch.chdir(tmp_path)

    wizard.state.target_dir = ""
    wizard.controller.revalidate()

    assert wizard.state.validation_ok is False
    assert [f["message"] for f in wizard.state.findings] == [
        "No target directory — type an absolute path first."
    ]


def test_save_case_reports_a_scaffold_failure(tmp_path, seed_transport_defaults, wizard):
    # A case root the scaffold cannot write into (here: an `Allrun` directory in the
    # way, standing in for a read-only root). The configs are written, but the wizard
    # must not land on Review still telling the user to click Save.
    (tmp_path / "Allrun").mkdir()
    seed_transport_defaults(wizard)
    wizard.state.target_dir = str(tmp_path)

    wizard.controller.save_case()  # must not raise

    assert wizard.state.current_step == "review"
    assert wizard.state.validation_ok is False  # not None → Review shows the failure
    assert wizard.state.scaffolded == []
    assert "error" in wizard.state.save_report
    assert wizard.state.findings


def test_failed_scan_clears_the_mesh_written_alert(tmp_path, wizard):
    _scan_tube_bank(wizard, tmp_path)
    wizard.controller.write_mesh()
    assert wizard.state.mesh_written  # the green "Wrote: …" alert is up

    wizard.state.stl_dir = str(tmp_path / "does_not_exist")
    asyncio.run(wizard.controller.load_geometry())

    # The failure must not sit above a stale success alert, and must not read as info.
    assert wizard.state.geometry_patches == []
    assert wizard.state.mesh_written == []
    assert wizard.state.geometry_severity == "error"
    assert "Could not read geometry" in wizard.state.geometry_status


def _slow_scan(spec, scanned: list[str], seconds: float = _SCAN_SECONDS):
    """Stand-in for reading a large STL: blocks for ``seconds``, then yields ``spec``."""

    def scan(path, **_kwargs):
        scanned.append(str(path))
        time.sleep(seconds)
        return spec

    return scan


def test_scan_keeps_the_event_loop_running(monkeypatch, heartbeat_ticks, wizard):
    # trame is single-threaded, so an STL read on the loop freezes the whole UI —
    # every other client callback, including the heartbeat, stops for its duration.
    wizard.state.stl_dir = str(_TRI_SURFACE)
    monkeypatch.setattr(
        "neofoam.ui.geometry_panel.discover_geometry",
        _slow_scan(discover_geometry(_TRI_SURFACE), []),
    )

    ticks = heartbeat_ticks(wizard.controller.load_geometry)

    assert ticks >= 5  # ~15 over a 0.3 s scan; 0 while the loop is blocked
    assert wizard.state.geometry_patches  # and the scan still landed


def test_scan_is_busy_while_it_runs(monkeypatch, wizard):
    # Without a busy flag the Scan button looks idle through the whole freeze.
    wizard.state.stl_dir = str(_TRI_SURFACE)
    spec = discover_geometry(_TRI_SURFACE)
    busy_while_scanning: list[bool] = []
    monkeypatch.setattr(
        "neofoam.ui.geometry_panel.discover_geometry",
        lambda *_a, **_kw: (busy_while_scanning.append(wizard.state.geometry_busy), spec)[1],
    )

    asyncio.run(wizard.controller.load_geometry())

    assert busy_while_scanning == [True]
    assert wizard.state.geometry_busy is False


def test_scan_started_while_one_runs_is_dropped(monkeypatch, wizard):
    # Clicks queued during the freeze all land once it ends; a second scan would
    # re-seed the boundary-condition forms underneath the first one's results.
    wizard.state.stl_dir = str(_TRI_SURFACE)
    scanned: list[str] = []
    monkeypatch.setattr(
        "neofoam.ui.geometry_panel.discover_geometry",
        _slow_scan(discover_geometry(_TRI_SURFACE), scanned),
    )

    async def drive() -> None:
        first = asyncio.create_task(wizard.controller.load_geometry())
        await asyncio.sleep(_SCAN_SECONDS / 3)  # the first scan is in flight
        await wizard.controller.load_geometry()  # a click queued during it
        await first

    asyncio.run(drive())

    assert scanned == [str(_TRI_SURFACE)]
    assert wizard.state.geometry_busy is False


def test_revalidate_reruns_without_resaving(tmp_path, seed_transport_defaults, wizard):
    seed_transport_defaults(wizard)
    wizard.state.target_dir = str(tmp_path)
    wizard.controller.save_case()

    allrun_mtime = (tmp_path / "Allrun").stat().st_mtime
    wizard.controller.revalidate()  # must not raise, must not re-scaffold
    assert (tmp_path / "Allrun").stat().st_mtime == allrun_mtime
    assert wizard.state.validation_ok is not None


def test_pick_one_family_starts_on_one_member_and_switching_deselects_the_other(wizard):
    state, ctrl = wizard.state, wizard.controller

    # Pimple and Simple want contradictory ddtSchemes — exactly one is ever selected.
    assert state.choice_PressureVelocityAlgorithm == "Pimple"
    assert state.sel_Pimple is True
    assert state.sel_Simple is False

    ctrl.select_model("Simple")

    assert state.choice_PressureVelocityAlgorithm == "Simple"
    assert state.sel_Pimple is False
    assert state.sel_Simple is True


def test_save_case_writes_only_the_chosen_algorithm(tmp_path, wizard, seed_transport_defaults):
    seed_transport_defaults(wizard)

    wizard.state.target_dir = str(tmp_path / "pimple")
    wizard.controller.save_case()
    pimple = (tmp_path / "pimple" / "system" / "fvSolution").read_text()
    assert "PIMPLE" in pimple
    assert "SIMPLE" not in pimple  # the unselected algorithm's block is not written

    wizard.controller.select_model("Simple")
    wizard.state.target_dir = str(tmp_path / "simple")
    wizard.controller.save_case()
    simple = (tmp_path / "simple" / "system" / "fvSolution").read_text()
    assert "SIMPLE" in simple
    assert "PIMPLE" not in simple


def test_save_case_writes_no_piso_block_beside_pimple(tmp_path, seed_transport_defaults, wizard):
    # A PISO block is only read when no PIMPLE block exists, so writing both left
    # the user a block (and two wizard panels) whose edits never took effect.
    seed_transport_defaults(wizard)

    wizard.state.target_dir = str(tmp_path)
    wizard.controller.save_case()

    fv_solution = (tmp_path / "system" / "fvSolution").read_text()
    assert "PIMPLE" in fv_solution
    assert "PISO" not in fv_solution


def test_save_case_writes_a_hand_added_scheme_entry_as_openfoam_tokens(
    tmp_path, wizard, seed_transport_defaults
):
    # The Numerics step's "+ add entry" puts a key the schema does not declare into a
    # scheme section, as the same structured object its declared siblings hold.
    schemes = wizard.state.form_pimple_fv_schemes
    seed_transport_defaults(wizard)
    wizard.state.form_pimple_fv_schemes = {
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

    wizard.state.target_dir = str(tmp_path)
    wizard.controller.save_case()

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
def test_selecting_a_turbulence_model_rewrites_turbulence_properties(model, expected, wizard):
    wizard.controller.select_model(model)

    assert _turbulence_properties(wizard) == expected


def test_selecting_another_family_leaves_turbulence_properties_alone(wizard):
    wizard.controller.select_model("laminar")

    wizard.controller.select_model("Simple")

    assert _turbulence_properties(wizard) == {"simulationType": "laminar"}


def test_panel_chip_shows_the_owning_models_label(pristine_wizard):
    template = pristine_wizard.state["trame__template_main"]
    assert "Adaptive time step (Courant)\n</VChip>" in template
    assert "\ncourant\n</VChip>" not in template


def test_save_case_writes_the_selected_turbulence_model(tmp_path, wizard, seed_transport_defaults):
    seed_transport_defaults(wizard)
    wizard.controller.select_model("kOmegaSST")
    wizard.state.target_dir = str(tmp_path)

    wizard.controller.save_case()

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


def test_boundary_conditions_step_explains_its_empty_state(pristine_wizard):
    # Before a scan each field panel is a bare "Property Name [+]" row; the step says
    # where patches come from and that the row adds one by hand, until a scan ran.

    template = pristine_wizard.state["trame__template_main"]
    assert "run Scan in the Geometry step to seed them" in template
    assert 'v-show="!geometry_patches.length"' in template
    # The adder row is labelled "Patch name" itself, so the hint need not name the box.
    assert "Property Name" not in template


def test_boundary_conditions_hint_yields_to_patches_in_the_forms(pristine_wizard):
    # "Load case" fills the forms with the case's patches without a scan, so the hint
    # also asks the forms; the patch count of each is a term of its condition.

    template = pristine_wizard.state["trame__template_main"]
    alert = template.split('v-show="!geometry_patches.length"')[1].split(">")[0]
    assert alert.split('v-if="')[1].split('"')[0] == (
        "!(Object.keys(form_u_field_config__bc.boundaryField || {}).length"
        " || Object.keys(form_p_field_config__bc.boundaryField || {}).length"
        " || Object.keys(form_p_rgh_field_config__bc.boundaryField || {}).length"
        " || Object.keys(form_t_field_config__bc.boundaryField || {}).length"
        " || Object.keys(form_alphat_field_config__bc.boundaryField || {}).length)"
    )


def test_forms_receive_the_adder_translations(pristine_wizard):
    from neofoam.ui.form_schema import ADDER_TRANSLATIONS  # noqa: PLC0415

    assert ':translations="form_translations"' in pristine_wizard.state["trame__template_main"]
    assert pristine_wizard.state["form_translations"] == ADDER_TRANSLATIONS


@pytest.mark.parametrize("location", ["left", "right"])
def test_drawers_are_overlays_below_the_desktop_breakpoint(location, pristine_wizard):
    # A permanent 300 px step drawer (plus the 400 px AI drawer) leaves a phone no
    # room for the forms: below Vuetify's md breakpoint both become temporary overlays.

    template = pristine_wizard.state["trame__template_main"]
    drawer = template.split(f'location="{location}"')[1].split(">")[0]
    assert ':temporary="$vuetify.display.smAndDown"' in drawer
    assert ':permanent="!$vuetify.display.smAndDown"' in drawer


def test_drawers_start_closed_on_a_phone(pristine_wizard):
    # The overlays read their own open flags, so the desktop's open-by-default
    # drawers never cover a phone screen on load.

    template = pristine_wizard.state["trame__template_main"]
    assert pristine_wizard.state.main_drawer_mobile is False
    assert pristine_wizard.state.ai_panel_mobile is False
    assert ':modelValue="$vuetify.display.smAndDown ? main_drawer_mobile : main_drawer"' in template
    assert ':modelValue="$vuetify.display.smAndDown ? ai_panel_mobile : ai_panel"' in template


def test_picking_a_step_closes_the_phone_drawer(pristine_wizard):
    template = pristine_wizard.state["trame__template_main"]
    assert "@click=\"current_step = 'bcs'; main_drawer_mobile = false\"" in template


def test_target_directory_moves_to_a_second_toolbar_row_on_a_phone(pristine_wizard):
    template = pristine_wizard.state["trame__template_main"]
    extension = template.split('<template v-if="$vuetify.display.smAndDown" v-slot:extension>')[1]
    assert 'v-model="target_dir"' in extension.split("</template>")[0]


def test_load_case_button_sits_beside_the_target_directory_on_both_toolbar_rows(pristine_wizard):
    template = pristine_wizard.state["trame__template_main"]
    desktop, phone = template.split('<template v-if="$vuetify.display.smAndDown" v-slot:extension>')
    for row in (desktop, phone.split("</template>")[0]):
        field, _, after = row.partition('v-model="target_dir"')
        assert "Load case" in after.split("Save case")[0]


def test_load_case_button_needs_a_target_directory(pristine_wizard):
    template = pristine_wizard.state["trame__template_main"]
    button = template.split("Load case")[0].rsplit("<", 1)[1]
    assert ':disabled="!target_dir.trim() || ai_busy"' in button


def test_load_case_button_loads_the_target_case_without_an_api_key(monkeypatch, wizard):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    wizard.state.target_dir = str(_PITZ_DAILY)

    wizard.controller.load_target_case()

    transport = next(
        e for e in wizard.controller.get_entries() if e.config_name == "transport_properties_config"
    )
    assert wizard.state[transport.state_key] == {"transportModel": "Newtonian", "nu": 1e-05}


def test_chat_renders_assistant_messages_as_html_and_user_messages_as_text(pristine_wizard):
    template = pristine_wizard.state["trame__template_main"]
    user, assistant = (bubble for bubble in template.split("<div") if 'v-if="m.role ' in bubble)
    assert "m.role === 'user'" in user
    assert "{{ m.content }}" in user
    assert "white-space: pre-wrap" in user
    assert 'v-html="chat_html[i]"' in assistant
    assert "{{" not in assistant
    for bubble in (user, assistant):
        assert "overflow-wrap: anywhere" in bubble
