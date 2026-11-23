# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The flat NeoN solver blocks through the real "Load case" button in headless Chromium.

Opt-in with ``pytest -m browser``. A form that keeps its previous arm across a load, or
drops a value on a solver switch, only shows in a mounted page. Assertions read
``window.trame.state``.
"""

from __future__ import annotations

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

_CASES = Path(__file__).parent / "cases"
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


# One server per test: its trame state outlives a page, so a test would see the last edit.
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
        opened = browser.new_page(viewport={"width": 1500, "height": 1100})
        opened.set_default_timeout(15_000)
        opened.goto(wizard_url)
        opened.wait_for_function("window.trame?.state?.get('current_step') !== undefined")
        yield opened
        browser.close()


def _load_block_row(page: Any, case: Path) -> Any:
    """Load ``case`` through the toolbar and return the row of the ``p`` block."""
    page.get_by_role("button", name="Load case").first.click()
    dialog = page.locator(".v-dialog.v-overlay--active")
    expect(dialog).to_be_visible()
    page.evaluate("path => window.trame.state.set('picker_dir', path)", str(case))
    expect(dialog).to_contain_text(case.name)
    dialog.get_by_role("button", name="Load this directory").click()
    page.wait_for_function("path => window.trame.state.get('target_dir') === path", arg=str(case))
    # The load remounts the form, which would close a dropdown opened before it lands.
    page.wait_for_function("() => window.trame.state.get('form_epoch') > 0")
    page.locator("nav.v-navigation-drawer").first.get_by_text("Numerics", exact=True).click()
    page.get_by_text("fvSolution · PimpleNeoN").first.click()
    page.locator(".nf-field-head", has=page.locator(".nf-field-name", has_text="p")).last.click()
    rows = page.locator(".nf-union-row")
    expect(rows).to_have_count(4)
    return rows.filter(has=page.locator(".nf-compact-key", has_text="p*")).first


def _block(page: Any) -> dict[str, Any]:
    return page.evaluate("key => window.trame.state.get(key).solvers.p", _SOLUTION)


def _pick_solver(page: Any, row: Any, label: str, name: str | None = None) -> None:
    name = name or label
    option = page.get_by_role("option", name=label, exact=True)
    # A forced click misses while the expansion panel is still moving, so it is repeated.
    for _ in range(10):
        row.locator(".v-select .v-field").first.click(force=True)
        try:
            option.wait_for(timeout=1_000)
            break
        except sync_api.TimeoutError:
            continue
    option.click()
    page.wait_for_function(
        "([key, name]) => window.trame.state.get(key).solvers.p.solver === name",
        arg=[_SOLUTION, name],
    )


def test_load_case_shows_a_flat_block_by_its_solver_name(page, tmp_path):
    case = shutil.copytree(_CASES / "neon_typed_block", tmp_path / "typed_case")

    row = _load_block_row(page, case)

    expect(row.locator(".v-select__selection-text")).to_have_text(["Cg", "Ic"])


def test_solver_switch_keeps_the_shared_values(page, tmp_path):
    case = shutil.copytree(_CASES / "neon_typed_block", tmp_path / "typed_case")
    row = _load_block_row(page, case)

    _pick_solver(page, row, "Gmres")

    assert _block(page) == {
        "solver": "Gmres",
        "preconditioner": {"type": "Ic"},
        "tolerance": 1e-07,
        "relTol": 0.01,
        "maxIter": 500,
    }


def test_solver_switch_opens_no_dialog(page, tmp_path):
    case = shutil.copytree(_CASES / "neon_typed_block", tmp_path / "typed_case")
    row = _load_block_row(page, case)

    _pick_solver(page, row, "Bicg")

    expect(page.locator(".v-dialog.v-overlay--active")).to_have_count(0)


def test_config_file_fallback_shows_its_path_and_the_shared_fields(page, tmp_path):
    case = shutil.copytree(_CASES / "neon_typed_block", tmp_path / "typed_case")
    row = _load_block_row(page, case)

    _pick_solver(page, row, "Config file", "Ginkgo")
    row.get_by_role("button", name="All entries").click()

    expect(row.locator(".v-field-label:visible")).to_have_text(
        [
            "Solver",
            "JSON file (relative to the case directory)*",
            "Tolerance",
            "Rel Tol",
            "Max Iter",
            "Min Iter",
        ]
    )
