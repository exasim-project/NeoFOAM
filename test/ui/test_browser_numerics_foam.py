# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The per-field Numerics step of ``incompressibleFluid`` through the real form and Save."""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
from .test_browser_forms import _launch_chromium  # noqa: E402

expect = sync_api.expect

pytestmark = pytest.mark.browser

_TEST_DIR = Path(__file__).parents[1]
_RUNNABLE_CASE = _TEST_DIR / "setup_pimple"
_NUMERICS = Path(__file__).parent / "cases" / "foam_pcg_gamg" / "system"
_SERVE_WIZARD = (
    "import sys; from neofoam.ui import build_app; "
    "build_app(solver_name='incompressibleFluid').start("
    "port=int(sys.argv[1]), open_browser=False, show_connection_info=False)"
)
_SOLUTION = "form_pimple_fv_solution"
_U_BLOCK = """    U
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-09;
        relTol          0.01;
    }
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def wizard_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    port = _free_port()
    cwd = tmp_path_factory.mktemp("wizard")
    server = subprocess.Popen([sys.executable, "-c", _SERVE_WIZARD, str(port)], cwd=cwd)
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.terminate()
        server.wait(timeout=10)


@pytest.fixture
def page(wizard_url: str) -> Iterator[Any]:
    with sync_api.sync_playwright() as playwright:
        browser = _launch_chromium(playwright)
        opened = browser.new_page(viewport={"width": 1600, "height": 1200})
        opened.set_default_timeout(15_000)
        opened.goto(wizard_url)
        opened.wait_for_function("window.trame?.state?.get('current_step') !== undefined")
        yield opened
        browser.close()


@pytest.fixture
def case(tmp_path: Path) -> Path:
    target = shutil.copytree(_RUNNABLE_CASE, tmp_path / "case")
    for name in ("fvSolution", "fvSchemes"):
        shutil.copyfile(_NUMERICS / name, target / "system" / name)
    return target


def _replace_u_block(case: Path, block: str) -> None:
    path = case / "system/fvSolution"
    text = path.read_text()
    assert _U_BLOCK in text
    path.write_text(text.replace(_U_BLOCK, block))


def _load(page: Any, target: Path) -> None:
    page.get_by_role("button", name="Load case").first.click()
    dialog = page.locator(".v-dialog.v-overlay--active")
    expect(dialog).to_be_visible()
    page.evaluate("path => window.trame.state.set('picker_dir', path)", str(target))
    expect(dialog).to_contain_text(target.name)
    dialog.get_by_role("button", name="Load this directory").click()
    page.wait_for_function("() => window.trame.state.get('form_epoch') > 0")
    page.locator("nav.v-navigation-drawer").first.get_by_text("Numerics", exact=True).click()
    page.get_by_text("fvSolution · Pimple", exact=True).first.click()


def _field(page: Any, name: str) -> Any:
    label = page.locator(".nf-field-name:visible").get_by_text(name, exact=True)
    field = page.locator(".nf-field").filter(has=label).last
    head = field.locator("> .nf-field-head")
    if head.get_attribute("aria-expanded") == "false":
        head.click()
    return field


def _row(page: Any, name: str) -> Any:
    return _field(page, name).locator(".nf-union-row").first


def _text_field(row: Any, label: str) -> Any:
    # By the visible label: Vuetify doubles it in the accessible name.
    title = re.compile(rf"^\s*{re.escape(label)}\s*$")
    return row.locator(".v-input", has=row.page.locator("label", has_text=title)).locator("input")


def _options(page: Any, select: Any) -> list[str]:
    select.locator(".v-field").first.click()
    menu = page.locator(".v-overlay--active .v-list").last
    expect(menu).to_be_visible()
    names = [text.strip() for text in menu.get_by_role("option").all_inner_texts()]
    page.keyboard.press("Escape")
    expect(menu).to_be_hidden()
    return names


def _block(page: Any, name: str) -> dict[str, Any]:
    return page.evaluate("([k, n]) => window.trame.state.get(k).solvers[n]", [_SOLUTION, name])


def _save(page: Any) -> None:
    page.evaluate("() => window.trame.state.set('save_report', null)")
    page.get_by_role("button", name="Save case").first.click()
    page.wait_for_function("() => window.trame.state.get('save_report') !== null")
    report = page.evaluate("() => window.trame.state.get('save_report')")
    assert "error" not in report, report["error"]
    page.locator("nav.v-navigation-drawer").first.get_by_text("Numerics", exact=True).click()


def _preview(page: Any, file: str) -> str:
    return str(page.evaluate("f => window.trame.state.get('file_preview')[f]", file))


def _await_preview(page: Any, text: str) -> None:
    page.wait_for_function(
        "t => window.trame.state.get('file_preview')['system/fvSolution'].includes(t)", arg=text
    )


def test_backend_note_names_openfoam_without_a_conversion(page, case):
    _load(page, case)

    note = page.locator(".nf-backend:visible")

    expect(note).to_contain_text("Backend OpenFOAM")
    expect(note).not_to_contain_text("equivalents")


def test_pcg_with_nested_gamg_edits_and_saves_in_the_native_form(page, case):
    _load(page, case)
    row = _row(page, "p")
    expect(_field(page, "p").locator("> .nf-field-head")).to_contain_text("PCG · GAMG · tol 1e-09")

    row.get_by_role("button", name="All entries").click()
    expect(row.locator(".nf-all-heading")).to_contain_text(
        ["GAMG: Agglomeration", "GAMG: Sweeps", "GAMG: Correction", "GAMG: Smoothing"]
    )
    expect(_text_field(row, "N Cells In Coarsest Level")).to_have_value("50")
    _text_field(row, "N Vcycles").fill("3")
    _await_preview(page, "nVcycles")
    _save(page)

    saved = (case / "system/fvSolution").read_text()
    assert re.search(r"preconditioner\s*\{\s*preconditioner\s+GAMG;", saved)
    assert "nVcycles        3;" in saved and "nCellsInCoarsestLevel 50;" in saved
    assert "nPostSweeps" not in saved and "agglomerator" not in saved


@pytest.mark.parametrize(
    ("field", "solver_hidden", "preconditioner_hidden"),
    [
        ("p", {"PBiCG"}, {"DILU", "distributedDILU"}),
        ("U", {"PCG", "PPCG", "PPCR", "FPCG"}, {"DIC", "FDIC", "distributedDIC"}),
    ],
)
def test_each_field_offers_only_the_choices_its_matrix_accepts(
    page, case, field, solver_hidden, preconditioner_hidden
):
    _load(page, case)
    row = _row(page, field)

    solvers = _options(page, row.locator(".v-select").first)
    preconditioners = _options(page, row.locator(".v-select").nth(1))

    assert "Other" in solvers and not solver_hidden & set(solvers)
    assert preconditioners and not preconditioner_hidden & set(preconditioners)


def test_other_block_stays_editable_and_round_trips(page, case):
    _replace_u_block(
        case,
        """    U
    {
        solver          petsc;
        petsc
        {
            options
            {
                ksp_type        bicg;
            }
        }
        tolerance       1e-09;
    }
""",
    )
    _load(page, case)
    row = _row(page, "U")
    expect(_text_field(row, "Solver Name")).to_have_value("petsc")

    _text_field(row, "Tolerance").fill("2e-8")
    _await_preview(page, "2e-08")
    _save(page)

    saved = (case / "system/fvSolution").read_text()
    assert re.search(r"U\s*\{\s*solver\s+petsc;", saved)
    assert "ksp_type        bicg;" in saved and "solverName" not in saved
    assert re.search(r"tolerance\s+2e-08;", saved)


def test_key_no_plugin_models_is_reachable_and_survives_save(page, case):
    _replace_u_block(case, _U_BLOCK.replace("relTol          0.01;", "relTol 0.01; myKey 7;"))
    _load(page, case)
    row = _row(page, "U")

    row.get_by_role("button", name="All entries").click()

    expect(row.locator(".nf-all-heading", has_text=re.compile("^Other keys$"))).to_be_visible()
    expect(_text_field(row, "myKey")).to_have_value("7")
    _save(page)
    assert re.search(r"myKey\s+7;", (case / "system/fvSolution").read_text())


def test_regex_key_file_is_left_byte_identical(page, case):
    _replace_u_block(case, _U_BLOCK.replace("    U\n", '    "(U|k)"\n'))
    path = case / "system/fvSolution"
    original = path.read_bytes()
    _load(page, case)
    page.wait_for_function("() => (window.trame.state.get('rejected_slices') || []).length > 0")
    page.wait_for_function(
        "t => window.trame.state.get('file_preview')['system/fvSolution'] === t",
        arg=original.decode(),
    )

    page.get_by_role("button", name="Save case").first.click()
    page.wait_for_function("() => window.trame.state.get('save_report') !== null")

    assert path.read_bytes() == original


def test_preview_equals_the_saved_files(page, case):
    _load(page, case)
    row = _row(page, "p")
    row.get_by_role("button", name="All entries").click()
    _text_field(row, "N Pre Sweeps").fill("2")
    _await_preview(page, "nPreSweeps")

    expected = {file: _preview(page, file) for file in ("system/fvSolution", "system/fvSchemes")}
    _save(page)

    for file, text in expected.items():
        assert (case / file).read_text() == text


def test_same_as_switch_starts_on_only_for_identical_blocks(page, case):
    _load(page, case)

    switch = _field(page, "U").locator(".nf-final-head").get_by_role("checkbox")

    expect(switch).not_to_be_checked()
    assert _block(page, "U")["relTol"] != _block(page, "UFinal")["relTol"]


def _open_schemes(page: Any) -> None:
    page.locator("nav.v-navigation-drawer").first.get_by_text("Numerics", exact=True).click()
    page.get_by_text("fvSchemes · Pimple", exact=True).first.click()


def _schemes(page: Any) -> dict[str, Any]:
    return page.evaluate("() => window.trame.state.get('form_pimple_fv_schemes')")


def _pick(page: Any, select: Any, option: str) -> None:
    select.locator(".v-field").first.click()
    page.locator(".v-overlay--active .v-list").last.get_by_role(
        "option", name=option, exact=True
    ).click()


def _add_entry(section: Any, name: str) -> None:
    section.get_by_role("button", name="Add entry", exact=True).click()
    box = section.locator(".nf-compact-adder").get_by_role("textbox")
    box.fill(name)
    box.press("Enter")


def _await_scheme(page: Any, key: str, want: dict[str, Any]) -> None:
    page.wait_for_function(
        """([key, want]) => JSON.stringify(
            window.trame.state.get('form_pimple_fv_schemes').divSchemes[key]
        ) === JSON.stringify(want)""",
        arg=[key, want],
    )


def test_scheme_variant_switch_in_a_field_row_seeds_the_new_arm(page):
    _open_schemes(page)
    row = _field(page, "U").locator(".nf-compact-row", has_text="div(phi,U)").first
    interpolation = row.locator(".v-select").nth(1)

    _pick(page, interpolation, "limitedLinear")
    _await_scheme(page, "div(phi,U)", {"type": "Gauss", "interpolation": {"type": "limitedLinear"}})
    _text_field(row, "Coefficient*").fill("0.5")

    want = {"type": "Gauss", "interpolation": {"type": "limitedLinear", "coefficient": 0.5}}
    _await_scheme(page, "div(phi,U)", want)


def test_added_scheme_entry_returns_focus_and_can_be_deleted(page):
    _open_schemes(page)
    before = _schemes(page)["divSchemes"]
    section = _field(page, "Other entries").locator(".nf-compact", has_text="divSchemes").first

    _add_entry(section, "div(phi,k)")

    expect(section.get_by_role("button", name="Add entry", exact=True)).to_be_focused()
    page.wait_for_function(
        "() => 'div(phi,k)' in window.trame.state.get('form_pimple_fv_schemes').divSchemes"
    )
    section.get_by_role("button", name="Delete div(phi,k)", exact=True).click()
    page.wait_for_function(
        "() => !('div(phi,k)' in window.trame.state.get('form_pimple_fv_schemes').divSchemes)"
    )
    assert _schemes(page)["divSchemes"] == before


def test_name_of_a_field_entry_cannot_be_added_over_it(page):
    _open_schemes(page)
    before = _schemes(page)["divSchemes"]
    section = _field(page, "Other entries").locator(".nf-compact", has_text="divSchemes").first

    _add_entry(section, "div(phi,U)")

    adder = section.locator(".nf-compact-adder")
    expect(adder.get_by_role("alert")).to_contain_text("'div(phi,U)' already defined")
    expect(adder.get_by_role("button", name="Add", exact=True)).to_be_disabled()
    assert _schemes(page)["divSchemes"] == before


def test_dotted_scheme_key_is_stored_as_a_literal_key(page):
    _open_schemes(page)
    key = "div(phi,alpha.water)"
    section = _field(page, "Other entries").locator(".nf-compact", has_text="divSchemes").first

    _add_entry(section, key)
    row = section.locator(".nf-compact-row", has_text=key).first
    _pick(page, row.locator(".v-select").first, "Gauss")
    _pick(page, row.locator(".v-select").nth(1), "vanLeer")

    _await_scheme(page, key, {"type": "Gauss", "interpolation": {"type": "vanLeer"}})
    assert "div(phi,alpha" not in _schemes(page)["divSchemes"]
