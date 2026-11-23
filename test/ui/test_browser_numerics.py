# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The per-field Numerics step through the real form and Save, in headless Chromium."""

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

_CASE = Path(__file__).parents[1] / "setup_pimple"
_SERVE_WIZARD = (
    "import sys; from neofoam.ui import build_app; "
    "build_app(solver_name='incompressibleFluidNeoN').start("
    "port=int(sys.argv[1]), open_browser=False, show_connection_info=False)"
)
_SOLUTION = "form_pimple_neo_n_fv_solution"


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
def case(page: Any, tmp_path: Path) -> Path:
    """A real case loaded through the toolbar, with the fvSolution panel open."""
    target = shutil.copytree(_CASE, tmp_path / "case")
    page.get_by_role("button", name="Load case").first.click()
    dialog = page.locator(".v-dialog.v-overlay--active")
    expect(dialog).to_be_visible()
    page.evaluate("path => window.trame.state.set('picker_dir', path)", str(target))
    expect(dialog).to_contain_text(target.name)
    dialog.get_by_role("button", name="Load this directory").click()
    page.wait_for_function("() => window.trame.state.get('form_epoch') > 0")
    page.locator("nav.v-navigation-drawer").first.get_by_text("Numerics", exact=True).click()
    page.get_by_text("fvSolution · PimpleNeoN").first.click()
    return target


def _field(page: Any, name: str) -> Any:
    label = page.locator(".nf-field-name:visible").get_by_text(name, exact=True)
    field = page.locator(".nf-field").filter(has=label).last
    head = field.locator("> .nf-field-head")
    if head.get_attribute("aria-expanded") == "false":
        head.click()
    return field


def _block(page: Any, name: str) -> dict[str, Any]:
    return page.evaluate("([k, n]) => window.trame.state.get(k).solvers[n]", [_SOLUTION, name])


def _text_field(row: Any, label: str) -> Any:
    # By the visible label: Vuetify doubles it in the accessible name.
    title = re.compile(rf"^\s*{re.escape(label)}\s*$")
    return row.locator(".v-input", has=row.page.locator("label", has_text=title)).locator("input")


def _pick(page: Any, select: Any, label: str) -> None:
    select.locator(".v-field").first.click()
    page.locator(".v-overlay--active .v-list").last.get_by_role(
        "option", name=label, exact=True
    ).click()
    expect(page.locator(".v-menu .v-overlay__content:visible")).to_have_count(0)
    expect(select.locator(".v-select__selection-text")).to_have_text(label)


def _save(page: Any) -> None:
    page.evaluate("() => window.trame.state.set('save_report', null)")
    page.get_by_role("button", name="Save case").first.click()
    page.wait_for_function("() => window.trame.state.get('save_report') !== null")
    report = page.evaluate("() => window.trame.state.get('save_report')")
    assert "error" not in report, report["error"]
    # Save moves on to the Review step.
    page.locator("nav.v-navigation-drawer").first.get_by_text("Numerics", exact=True).click()


def _preview(page: Any, file: str) -> str:
    return str(page.evaluate("f => window.trame.state.get('file_preview')[f]", file))


def _await_preview(page: Any, text: str) -> None:
    page.wait_for_function(
        "t => window.trame.state.get('file_preview')['system/fvSolution'].includes(t)", arg=text
    )


def _use_multigrid(page: Any) -> Any:
    field = _field(page, "p")
    row = field.locator(".nf-union-row").first
    _pick(page, row.locator(".v-select").first, "Cg")
    _pick(page, row.locator(".v-select").nth(1), "Multigrid")
    row.get_by_role("button", name="All entries").click()
    return row


def test_a_field_row_collapses_and_summarises_its_solver(page, case):
    field = _field(page, "p")
    head = field.locator("> .nf-field-head")

    expect(field.locator("> .nf-field-body")).to_be_visible()
    head.click()

    expect(field.locator("> .nf-field-body")).to_be_hidden()
    block = _block(page, "p")
    expect(field.locator("> .nf-field-head .nf-field-summary")).to_contain_text(block["solver"])
    expect(field.locator("> .nf-field-head .nf-field-summary")).to_contain_text("tol ")


def test_all_entries_follow_the_current_selection(page, case):
    row = _use_multigrid(page)

    expect(row.locator(".nf-all-heading")).to_have_text(
        ["Solver: Cg", "Multigrid: Hierarchy", "Multigrid: Smoothing", "Multigrid: Pgm coarsening"]
    )
    expect(_text_field(row, "Max Levels")).to_have_attribute("placeholder", "10")
    deterministic = row.locator(".v-select").filter(has_text="Deterministic")
    expect(deterministic.locator("input").first).to_have_attribute("placeholder", "false")

    _pick(page, row.locator(".v-select").nth(1), "Jacobi")

    expect(row.locator(".nf-all-heading")).to_have_text(["Solver: Cg", "Preconditioner: Jacobi"])


def test_switching_away_from_multigrid_on_a_solver_that_refuses_it(page, case):
    row = _use_multigrid(page)
    _text_field(row, "Max Levels").fill("8")
    page.wait_for_function(
        "k => window.trame.state.get(k).solvers.p.preconditioner.max_levels === 8", arg=_SOLUTION
    )

    _pick(page, row.locator(".v-select").first, "Bicgstab")

    page.wait_for_function(
        "k => window.trame.state.get(k).solvers.p.solver === 'Bicgstab'", arg=_SOLUTION
    )
    assert _block(page, "p")["preconditioner"]["type"] != "Multigrid"


def test_vector_field_does_not_offer_the_solver_its_model_refuses(page, case):
    row = _field(page, "U").locator(".nf-union-row").first

    row.locator(".v-select .v-field").first.click()

    expect(page.get_by_role("option", name="Bicgstab", exact=True)).to_be_visible()
    expect(page.get_by_role("option", name="Gmres", exact=True)).to_have_count(0)
    expect(page.get_by_role("option", name="Config file", exact=True)).to_be_visible()


def test_switch_to_the_config_file_does_not_block_save(page, case):
    row = _field(page, "p").locator(".nf-union-row").first

    _pick(page, row.locator(".v-select").first, "Config file")
    _text_field(row, "JSON file (relative to the case directory)*").fill("p.json")
    _await_preview(page, "configFile")
    _save(page)

    saved = (case / "system/fvSolution").read_text()
    assert "p.json" in saved
    assert _block(page, "p").keys() == {"solver", "configFile"}


def test_same_as_round_trip_writes_the_full_block(page, case):
    field = _field(page, "U")
    switch = field.locator(".nf-final-head").get_by_role("checkbox")
    if not switch.is_checked():
        switch.check()
    row = field.locator(".nf-union-row").first
    _text_field(row, "Tolerance").fill("3e-7")
    page.wait_for_function(
        "k => window.trame.state.get(k).solvers.UFinal.tolerance === 3e-7", arg=_SOLUTION
    )
    assert _block(page, "UFinal") == _block(page, "U")

    switch.uncheck()
    final = field.locator(".nf-union-row").nth(1)
    expect(final).to_be_visible()
    _text_field(final, "Tolerance").fill("4e-7")
    page.wait_for_function(
        "k => window.trame.state.get(k).solvers.UFinal.tolerance === 4e-7", arg=_SOLUTION
    )

    assert _block(page, "U")["tolerance"] == 3e-7
    _save(page)
    saved = (case / "system/fvSolution").read_text()
    assert "UFinal" in saved and "$U" not in saved and "4e-07" in saved


def test_preview_equals_the_saved_files(page, case):
    row = _use_multigrid(page)
    _text_field(row, "N Pre Sweeps").fill("2")
    _await_preview(page, "nPreSweeps")

    expected = {file: _preview(page, file) for file in ("system/fvSolution", "system/fvSchemes")}
    _save(page)

    for file, text in expected.items():
        assert (case / file).read_text() == text


def test_unset_options_are_not_written_and_a_cleared_one_leaves_the_file(page, case):
    row = _use_multigrid(page)
    _save(page)
    saved = (case / "system/fvSolution").read_text()
    assert "max_levels" not in saved and "coarsening" not in saved

    row = _field(page, "p").locator(".nf-union-row").first
    _text_field(row, "Max Levels").fill("8")
    _await_preview(page, "max_levels")
    _save(page)
    assert "max_levels" in (case / "system/fvSolution").read_text()

    _text_field(row, "Max Levels").fill("")
    page.wait_for_function(
        "k => !('max_levels' in window.trame.state.get(k).solvers.p.preconditioner)", arg=_SOLUTION
    )
    _save(page)

    assert "max_levels" not in (case / "system/fvSolution").read_text()


def test_set_options_reach_the_saved_file(page, case):
    row = _use_multigrid(page)
    _text_field(row, "N Cells In Coarsest Level").fill("50")
    _text_field(row, "Max Iterations").fill("20")
    _await_preview(page, "max_iterations")

    _save(page)

    saved = (case / "system/fvSolution").read_text()
    assert "nCellsInCoarsestLevel 50" in saved and "max_iterations" in saved
