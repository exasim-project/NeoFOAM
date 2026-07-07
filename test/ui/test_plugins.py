# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for the wizard step-plugin registry + weave."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from neofoam.mcp.registry import resolve_solver
from neofoam.ui.forms import build_forms
from neofoam.ui.plugins import (
    AT_START,
    ENTRY_POINT_GROUP,
    StepContext,
    discover_step_plugins,
)
from neofoam.ui.steps import build_steps


class _Plugin:
    """A minimal StepPlugin recording register/render calls."""

    def __init__(self, id: str, after: str | None = None) -> None:
        self.id = id
        self.label = id.title()
        self.icon = "mdi-test"
        self.caption = f"caption for {id}"
        self.after = after
        self.registered = 0
        self.rendered = 0

    def register(self, ctx: StepContext) -> None:
        self.registered += 1

    def render(self, ctx: StepContext) -> None:
        self.rendered += 1


def _solver_entries():
    solver = resolve_solver("incompressibleFluid")
    return solver, build_forms(solver)


def test_discover_returns_injected_plugins_unchanged():
    plugins = [_Plugin("b", after="a"), _Plugin("a")]
    # Ordering is build_steps' job; discovery preserves the injected order.
    assert [p.id for p in discover_step_plugins(plugins)] == ["b", "a"]


def test_discover_skips_plugin_whose_factory_raises(monkeypatch):
    def _boom() -> Any:
        raise ImportError("optional dep missing")

    good = _Plugin("good")

    class _EP:
        def __init__(self, name: str, factory: Any) -> None:
            self.name = name
            self._factory = factory

        def load(self) -> Any:
            return self._factory

    def _fake_entry_points(*, group: str):
        assert group == ENTRY_POINT_GROUP
        return [_EP("broken", _boom), _EP("good", lambda: good)]

    monkeypatch.setattr("neofoam.ui.plugins.metadata.entry_points", _fake_entry_points)
    out = discover_step_plugins()
    assert [p.id for p in out] == ["good"]  # broken one skipped, not fatal


def test_build_steps_weaves_plugin_after_anchor():
    solver = resolve_solver("incompressibleFluid")
    entries = build_forms(solver)
    plugin = _Plugin("cad", after="models")
    steps = build_steps(solver, entries, [plugin])
    ids = [s.id for s in steps]
    assert ids == [
        "models",
        "cad",  # slotted right after its anchor
        "geometry",
        "bcs",
        "initial",
        "schemes",
        "sweep",
        "review",
    ]
    assert {s.id: s for s in steps}["cad"].entry_keys == []  # bespoke, form-less


def test_build_steps_appends_plugin_with_unknown_anchor():
    solver = resolve_solver("incompressibleFluid")
    entries = build_forms(solver)
    steps = build_steps(solver, entries, [_Plugin("x", after="nope")])
    assert steps[-1].id == "x"


def test_build_steps_resolves_plugin_chain_regardless_of_input_order():
    # `b` is anchored on plugin `a`, `a` on the built-in `models`. Even when `b`
    # is discovered before `a`, the fixed-point weave must land `models, a, b`.
    solver, entries = _solver_entries()
    a = _Plugin("a", after="models")
    b = _Plugin("b", after="a")
    ids = [s.id for s in build_steps(solver, entries, [b, a])]
    assert ids[:3] == ["models", "a", "b"]


def test_build_steps_appends_plugin_with_no_anchor_at_end():
    solver, entries = _solver_entries()
    steps = build_steps(solver, entries, [_Plugin("tail", after=None)])
    assert steps[-1].id == "tail"


def test_build_steps_at_start_places_step_first():
    solver, entries = _solver_entries()
    ids = [s.id for s in build_steps(solver, entries, [_Plugin("cad", after=AT_START)])]
    assert ids[0] == "cad"
    assert ids[1] == "models"  # built-ins follow, in order


def test_build_steps_at_start_keeps_input_order_and_allows_anchoring():
    solver, entries = _solver_entries()
    # Two AT_START steps keep discovery order; a third anchors on the first.
    ids = [
        s.id
        for s in build_steps(
            solver,
            entries,
            [
                _Plugin("cad", after=AT_START),
                _Plugin("scan", after=AT_START),
                _Plugin("post-cad", after="cad"),
            ],
        )
    ]
    assert ids[:4] == ["cad", "post-cad", "scan", "models"]


def test_build_steps_two_plugins_on_same_anchor_keep_input_order():
    solver, entries = _solver_entries()
    ids = [
        s.id
        for s in build_steps(
            solver,
            entries,
            [_Plugin("p1", after="models"), _Plugin("p2", after="models")],
        )
    ]
    # Both anchor on models; insertion keeps discovery order right after it.
    assert ids[:3] == ["models", "p1", "p2"]


def test_build_app_registers_and_renders_plugin_step():
    pytest.importorskip("trame_flow")
    from trame.app import get_server  # type: ignore

    from neofoam.ui import build_app

    plugin = _Plugin("cad", after="models")

    def _register(ctx: StepContext) -> None:
        plugin.registered += 1
        ctx.server.state.cad_marker = "hello"

    plugin.register = _register  # type: ignore[method-assign]
    server = build_app(server=get_server("neofoam_ui_test_plugin"), plugins=[plugin])
    steps = server.controller.get_steps()
    assert [s.id for s in steps][:2] == ["models", "cad"]
    assert plugin.registered == 1  # register ran once during build
    assert plugin.rendered == 1  # render drew the panel once
    assert server.state.cad_marker == "hello"  # plugin seeded its own state


def test_step_context_carries_shared_handles():
    ctx = StepContext(
        server=SimpleNamespace(),
        solver=object(),
        solver_name="incompressibleFluid",
        entries=[],
        json_forms=type("JF", (), {}),
        v3=None,
        html=None,
        client=None,
        sweep=None,
        schema_key=lambda e: "k",
    )
    assert ctx.solver_name == "incompressibleFluid"
    assert ctx.extras == {}
