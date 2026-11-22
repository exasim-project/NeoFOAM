# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The bundled form renderers in a real browser — opt-in: ``pytest -m browser``.

Nothing in the headless suite mounts the JS bundle, so every bug these tests pin was
found by hand in a browser first. They are deselected by default (``conftest.py``)
because they need Chromium: Playwright's own build, else the newest one cached under
``~/.cache/ms-playwright`` (the fallback a venv newer than the cache depends on).

* The wizard runs in a subprocess (trame's server blocks, and one process holds one
  server), with a temporary working directory so nothing lands in the checkout.
* One server is shared per module; trame state lives on the server, so every test
  starts from a fresh page with the forms put back to their first-load values.
* Tests that judge a *whole* session (mounting dirties nothing, the console stays
  clean) walk a server of their own, once per solver.
* Assertions read the wizard's real state (``window.trame.state``), and waiting is
  Playwright's own (``expect`` / ``wait_for_function``), never a sleep.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

pytestmark = pytest.mark.browser

_SOLVERS = ["incompressibleFluid", "incompressibleFluidNeoN", "incompressibleVoF"]
#: trame's client probes an endpoint the server does not route; harmless and the
#: only console output a healthy session produces.
_KNOWN_CONSOLE = [
    "error: Failed to load resource: the server responded with a status of 405 (Method Not Allowed)"
]
_SERVE = (
    "import sys; from neofoam.ui import build_app; "
    "build_app(solver_name=sys.argv[1]).start("
    "port=int(sys.argv[2]), open_browser=False, show_connection_info=False)"
)
_CLOSED_PANELS = (
    ".v-expansion-panel:not(.v-expansion-panel--active) > .v-expansion-panel-title:visible"
)
_STATE_EQUALS = """([key, path, want]) => {
  const same = (a, b) => a === b || (!!a && !!b && typeof a === 'object' && typeof b === 'object'
    && Object.keys(a).length === Object.keys(b).length
    && Object.keys(a).every((k) => same(a[k], b[k])))
  return same(path.reduce((value, part) => value?.[part], window.trame.state.get(key)), want)
}"""
_FORM_STATE = """() => Object.fromEntries(Object.entries(window.trame.state.state)
  .filter(([key]) => key.startsWith('form_') && key !== 'form_translations'))"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(server: subprocess.Popen[bytes], port: int, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        try:  # the pause between two probes, and the check that the server still lives
            server.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            continue
        raise RuntimeError(f"the wizard exited with {server.returncode} before serving")
    raise TimeoutError(f"the wizard on port {port} never came up")


@contextmanager
def _wizard(solver: str, cwd: Path) -> Iterator[str]:
    """Serve ``solver``'s wizard from a subprocess; yields its URL."""
    port = _free_port()
    server = subprocess.Popen([sys.executable, "-c", _SERVE, solver, str(port)], cwd=cwd)
    try:
        _wait_for_port(server, port)
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def _launch_chromium(playwright: Any) -> Any:
    try:
        return playwright.chromium.launch()
    except sync_api.Error:
        pass
    cached = sorted(
        Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux64/chrome"),
        key=lambda chrome: int(chrome.parts[-3].rpartition("-")[2]),
    )
    if not cached:
        pytest.skip("no Chromium: run `playwright install chromium`")
    try:
        return playwright.chromium.launch(executable_path=str(cached[-1]))
    except sync_api.Error as exc:
        pytest.skip(f"Chromium does not launch: {exc}")


@pytest.fixture(scope="module")
def chromium() -> Iterator[Any]:
    with sync_api.sync_playwright() as playwright:
        browser = _launch_chromium(playwright)
        yield browser
        browser.close()


def _open(chromium: Any, url: str, console: list[str], viewport: tuple[int, int]) -> Any:
    """A fresh page on the wizard, its console errors and page errors going to ``console``."""
    page = chromium.new_page(viewport={"width": viewport[0], "height": viewport[1]})
    page.set_default_timeout(15_000)
    page.on(
        "console",
        lambda message: (
            console.append(f"{message.type}: {message.text}") if message.type == "error" else None
        ),
    )
    page.on("pageerror", lambda error: console.append(f"pageerror: {error}"))
    page.goto(url)
    page.wait_for_function("window.trame?.state?.get('current_step') !== undefined")
    return page


@pytest.fixture(scope="module")
def fluid(chromium: Any, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """The shared ``incompressibleFluid`` wizard: its URL and its first-load form state."""
    with _wizard("incompressibleFluid", tmp_path_factory.mktemp("fluid")) as url:
        page = _open(chromium, url, [], (1600, 1000))
        pristine = page.evaluate(_FORM_STATE)
        page.close()
        yield {"url": url, "pristine": pristine}


def _fresh_page(chromium: Any, fluid: dict[str, Any], viewport: tuple[int, int]) -> Iterator[Any]:
    console: list[str] = []
    page = _open(chromium, fluid["url"], console, viewport)
    page.evaluate(
        "forms => Object.entries(forms).forEach(([k, v]) => window.trame.state.set(k, v))",
        fluid["pristine"],
    )
    yield page
    page.close()
    assert [line for line in console if line not in _KNOWN_CONSOLE] == []


@pytest.fixture
def page(chromium: Any, fluid: dict[str, Any]) -> Iterator[Any]:
    yield from _fresh_page(chromium, fluid, (1600, 1000))


@pytest.fixture
def phone(chromium: Any, fluid: dict[str, Any]) -> Iterator[Any]:
    yield from _fresh_page(chromium, fluid, (400, 850))


def _state(page: Any, key: str) -> Any:
    return page.evaluate("key => window.trame.state.get(key)", key)


def _set_state(page: Any, key: str, value: Any) -> None:
    page.evaluate("([key, value]) => window.trame.state.set(key, value)", [key, value])


def _await_state(page: Any, key: str, path: list[str], want: Any) -> Any:
    """The state under ``key``/``path`` once it equals ``want`` — or as it is after the wait."""
    try:
        page.wait_for_function(_STATE_EQUALS, arg=[key, path, want], timeout=5_000)
    except sync_api.TimeoutError:
        pass  # the caller's assert then shows the difference
    value = _state(page, key)
    for part in path:
        value = (value or {}).get(part)
    return value


def _settle(page: Any) -> None:
    """Let the forms mounted so far render and report: two animation frames."""
    page.evaluate(
        "() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))"
    )


def _goto_step(page: Any, label: str) -> None:
    drawer = page.locator("nav.v-navigation-drawer").first
    if "v-navigation-drawer--active" not in (drawer.get_attribute("class") or ""):
        page.get_by_role("banner").get_by_role("button").first.click()
    drawer.get_by_text(label, exact=True).click()
    expect(page.locator(".nf-step-title:visible")).to_have_text(label)


def _open_panel(page: Any, step: str, title: str) -> Any:
    _goto_step(page, step)
    # The title may be followed by its owner-model chip, never by more of a longer title.
    heading = re.compile(rf"^\s*{re.escape(title)}(\s\s|\s*$)")
    page.locator(_CLOSED_PANELS, has_text=heading).first.click()
    panel = page.locator(".v-expansion-panel--active:visible", has_text=title).first
    expect(panel.locator(".v-expansion-panel-text")).to_be_visible()
    return panel


def _open_every_panel(page: Any) -> None:
    closed = page.locator(_CLOSED_PANELS)
    while remaining := closed.count():
        closed.first.click()
        expect(closed).to_have_count(remaining - 1)
    _settle(page)


def _pick(page: Any, scope: Any, label: str, option: str) -> None:
    # Forced: Vuetify lays the selection text over the (one character wide) input.
    scope.get_by_role("combobox", name=label, exact=True).click(force=True)
    page.get_by_role("option", name=option, exact=True).click()


def _add_entry(section: Any, button: str, name: str) -> None:
    section.get_by_role("button", name=button, exact=True).click()
    box = section.locator(".nf-compact-adder").get_by_role("textbox")
    box.fill(name)
    box.press("Enter")


def _solver_card(panel: Any, name: str) -> Any:
    return (
        panel.locator(".nf-card-entry").filter(has=panel.page.get_by_text(name, exact=True)).first
    )


_SCHEMES = "form_pimple_fv_schemes"
_SOLUTION = "form_pimple_fv_solution"
_U_PATCHES = "form_u_field_config__bc"
_TRANSPORT = "form_transport_properties_config"


def test_scheme_variant_switch_keeps_type_and_seeds_the_new_arm(page):
    panel = _open_panel(page, "Numerics", "fvSchemes · Pimple")
    row = panel.locator(".nf-compact-row", has_text="div(phi,U)").first
    path = ["divSchemes", "div(phi,U)"]

    _pick(page, row, "Interpolation*", "limitedLinear")

    switched = _await_state(
        page, _SCHEMES, path, {"type": "Gauss", "interpolation": {"type": "limitedLinear"}}
    )
    assert switched == {"type": "Gauss", "interpolation": {"type": "limitedLinear"}}
    expect(row.get_by_role("alert")).to_contain_text("is a required property")

    row.get_by_role("textbox", name="Coefficient*").fill("0.5")

    filled = {"type": "Gauss", "interpolation": {"type": "limitedLinear", "coefficient": 0.5}}
    assert _await_state(page, _SCHEMES, path, filled) == filled
    expect(row.get_by_role("alert")).to_have_count(0)

    _pick(page, row, "Interpolation*", "upwind")

    back = {"type": "Gauss", "interpolation": {"type": "upwind"}}
    assert _await_state(page, _SCHEMES, path, back) == back


def test_added_scheme_entry_returns_focus_and_can_be_deleted(page):
    panel = _open_panel(page, "Numerics", "fvSchemes · Pimple")
    section = panel.locator(".nf-compact", has_text="divSchemes").first
    before = _state(page, _SCHEMES)["divSchemes"]

    _add_entry(section, "Add entry", "div(phi,k)")

    expect(section.get_by_role("button", name="Add entry", exact=True)).to_be_focused()
    added = _await_state(page, _SCHEMES, ["divSchemes"], {**before, "div(phi,k)": {}})
    assert list(added) == [*before, "div(phi,k)"]

    section.get_by_role("button", name="Delete div(phi,k)", exact=True).click()

    assert _await_state(page, _SCHEMES, ["divSchemes"], before) == before


def test_duplicate_entry_name_shows_an_error_and_cannot_be_added(page):
    panel = _open_panel(page, "Numerics", "fvSchemes · Pimple")
    section = panel.locator(".nf-compact", has_text="divSchemes").first
    before = _state(page, _SCHEMES)["divSchemes"]

    _add_entry(section, "Add entry", "div(phi,U)")

    adder = section.locator(".nf-compact-adder")
    expect(adder.get_by_role("alert")).to_contain_text("'div(phi,U)' already defined")
    expect(adder.get_by_role("button", name="Add", exact=True)).to_be_disabled()
    assert _state(page, _SCHEMES)["divSchemes"] == before


def test_dotted_scheme_key_is_stored_as_a_literal_key(page):
    panel = _open_panel(page, "Numerics", "fvSchemes · Pimple")
    section = panel.locator(".nf-compact", has_text="divSchemes").first
    key = "div(phi,alpha.water)"

    _add_entry(section, "Add entry", key)
    row = section.locator(".nf-compact-row", has_text=key).first
    _pick(page, row, key, "Gauss")
    _pick(page, row, "Interpolation*", "vanLeer")

    want = {"type": "Gauss", "interpolation": {"type": "vanLeer"}}
    assert _await_state(page, _SCHEMES, ["divSchemes", key], want) == want
    schemes = _state(page, _SCHEMES)
    assert "div(phi,alpha" not in schemes["divSchemes"]
    assert "alpha" not in schemes


def test_hand_added_dotted_patch_is_seeded_editable_and_deletable(page):
    panel = _open_panel(page, "Boundary conditions", "U — boundary conditions")
    patches = ["boundaryField"]

    _add_entry(panel, "Add patch", "wall.left")

    seeded = {"wall.left": {"type": "noSlip"}}
    assert _await_state(page, _U_PATCHES, patches, seeded) == seeded

    row = panel.locator(".nf-compact-row", has_text="wall.left").first
    _pick(page, row, "wall.left", "zeroGradient")

    edited = {"wall.left": {"type": "zeroGradient"}}
    assert _await_state(page, _U_PATCHES, patches, edited) == edited

    panel.get_by_role("button", name="Delete wall.left", exact=True).click()

    assert _await_state(page, _U_PATCHES, patches, {}) == {}


def test_solver_of_the_other_family_swaps_its_companion_key(page):
    panel = _open_panel(page, "Numerics", "fvSolution · Pimple")
    card = _solver_card(panel, "p")

    _pick(page, card, "solver", "GAMG")

    swapped = {"solver": "GAMG", "tolerance": 1e-06, "relTol": 0.05}
    assert _await_state(page, _SOLUTION, ["solvers", "p"], swapped) == swapped
    expect(card.get_by_role("combobox", name="smoother*", exact=True)).to_have_value("")
    expect(card.get_by_role("alert")).to_contain_text("is a required property")

    _pick(page, card, "smoother*", "GaussSeidel")

    complete = {**swapped, "smoother": "GaussSeidel"}
    assert _await_state(page, _SOLUTION, ["solvers", "p"], complete) == complete
    expect(card.get_by_role("alert")).to_have_count(0)


def test_unknown_solver_name_deletes_no_key(page):
    panel = _open_panel(page, "Numerics", "fvSolution · Pimple")
    card = _solver_card(panel, "pFinal")
    solver = card.get_by_role("combobox", name="solver", exact=True)

    solver.fill("Ginkgo")
    solver.press("Tab")

    kept = {"solver": "Ginkgo", "preconditioner": "DIC", "tolerance": 1e-06, "relTol": 0}
    assert _await_state(page, _SOLUTION, ["solvers", "pFinal"], kept) == kept


def test_number_field_stores_a_number_and_displays_it_as_typed_in_a_case(page):
    panel = _open_panel(page, "Numerics", "fvSolution · Pimple")
    tolerance = _solver_card(panel, "U").get_by_role("textbox", name="tolerance", exact=True)

    tolerance.fill("0.00001")

    stored = _await_state(page, _SOLUTION, ["solvers", "U", "tolerance"], 1e-05)
    assert stored == 1e-05 and isinstance(stored, float)
    expect(tolerance).to_have_value("0.00001")
    tolerance.blur()
    expect(tolerance).to_have_value("1e-5")


def test_number_field_rejects_a_decimal_comma_and_keeps_the_last_value(page):
    panel = _open_panel(page, "Numerics", "fvSolution · Pimple")
    card = _solver_card(panel, "U")
    tolerance = card.get_by_role("textbox", name="tolerance", exact=True)

    tolerance.fill("1e-5")
    tolerance.fill("0,5")

    expect(card.get_by_role("alert")).to_contain_text("'0,5' is not a number")
    assert _await_state(page, _SOLUTION, ["solvers", "U", "tolerance"], 1e-05) == 1e-05


def test_number_field_leaves_the_keyboard_choice_to_the_device(page):
    # `inputmode=decimal` brings up a phone keypad without `e` and `-`: no `1e-5`.
    panel = _open_panel(page, "Numerics", "fvSolution · Pimple")
    tolerance = _solver_card(panel, "U").get_by_role("textbox", name="tolerance", exact=True)

    assert tolerance.get_attribute("inputmode") is None


def test_conditionally_required_nu_is_starred_and_reports_when_empty(page):
    panel = _open_panel(page, "Models", "transportProperties")

    expect(panel.get_by_role("textbox", name="Nu*", exact=True)).to_be_visible()
    required = panel.get_by_role("alert").filter(has_text="is a required property")
    expect(required).to_have_count(1)


def test_typing_into_nu_keeps_the_focus(page):
    # The star comes from a rewritten schema; a fresh schema object per edit would
    # restart the form and drop the focus after the first character.
    panel = _open_panel(page, "Models", "transportProperties")
    nu = panel.get_by_role("textbox", name="Nu*", exact=True)

    nu.press_sequentially("1e-5")

    expect(nu).to_be_focused()
    expect(nu).to_have_value("1e-5")
    assert _await_state(page, _TRANSPORT, ["nu"], 1e-05) == 1e-05


def test_nu_loses_its_star_for_a_non_newtonian_model(page):
    panel = _open_panel(page, "Models", "transportProperties")
    expect(panel.get_by_role("textbox", name="Nu*", exact=True)).to_be_visible()

    _set_state(page, _TRANSPORT, {"transportModel": "CrossPowerLaw"})

    expect(panel.get_by_role("textbox", name="Nu", exact=True)).to_be_visible()
    expect(panel.get_by_role("textbox", name="Nu*", exact=True)).to_have_count(0)


@pytest.fixture(scope="module", params=_SOLVERS)
def walked(
    request: pytest.FixtureRequest, chromium: Any, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[dict[str, Any]]:
    """A fresh wizard after every panel of every step was opened: state before/after, console."""
    console: list[str] = []
    with _wizard(request.param, tmp_path_factory.mktemp(request.param)) as url:
        page = _open(chromium, url, console, (1600, 1000))
        before = page.evaluate(_FORM_STATE)
        drawer = page.locator("nav.v-navigation-drawer").first
        for label in drawer.get_by_role("listitem").all_inner_texts():
            _goto_step(page, label)
            _open_every_panel(page)
        after = page.evaluate(_FORM_STATE)
        page.close()
    yield {"before": before, "after": after, "console": console}


def test_mounting_every_form_leaves_the_state_untouched(walked):
    assert walked["before"]  # guard: the form keys were found
    assert walked["after"] == walked["before"]


def test_walking_every_step_logs_only_the_known_resource_error(walked):
    assert walked["console"] == _KNOWN_CONSOLE


def test_phone_starts_with_both_drawers_closed(phone):
    expect(phone.locator(".v-navigation-drawer--active")).to_have_count(0)
    assert _state(phone, "main_drawer_mobile") is False
    assert _state(phone, "ai_panel_mobile") is False


def test_phone_picking_a_step_closes_the_drawer(phone):
    phone.get_by_role("banner").get_by_role("button").first.click()
    drawer = phone.locator("nav.v-navigation-drawer").first
    expect(drawer).to_have_class(re.compile("v-navigation-drawer--active"))

    drawer.get_by_text("Numerics", exact=True).click()

    expect(phone.locator(".v-navigation-drawer--active")).to_have_count(0)
    assert _state(phone, "current_step") == "schemes"


@pytest.mark.parametrize("step", ["Numerics", "Boundary conditions"])
def test_phone_step_does_not_scroll_sideways(phone, step):
    inlet = {"type": "fixedValue", "value": "uniform (1 0 0)"}
    _set_state(phone, _U_PATCHES, {"boundaryField": {"inlet": inlet}})
    _goto_step(phone, step)

    _open_every_panel(phone)

    widths = phone.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
    assert widths[0] <= widths[1]
