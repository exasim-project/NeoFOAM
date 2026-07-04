# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""AI chat controller tests with a network-free stub agent (no browser, no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pybFoam")
pytest.importorskip("trame")

from trame.app import get_server  # noqa: E402

from neofoam.agent.case_fill import build_case_output_model  # noqa: E402
from neofoam.framework.solver.configurations import configurations  # noqa: E402
from neofoam.mcp.registry import resolve_solver  # noqa: E402
from neofoam.ui.agent_panel import build_agent_panel  # noqa: E402
from neofoam.ui.forms import build_forms  # noqa: E402

_COUNTER = {"n": 0}


def _server() -> Any:
    _COUNTER["n"] += 1
    srv = get_server(f"neofoam_ui_agent_{_COUNTER['n']}")
    srv.state.target_dir = ""  # no auto-save in tests
    return srv


def _solver() -> Any:
    return resolve_solver("incompressibleFluid")


def _prebuilt_case_spec(solver: Any) -> Any:
    cfgs = configurations(solver)
    case_spec_cls = build_case_output_model(solver=solver)
    return case_spec_cls.model_construct(
        transport_properties_config=cfgs["TransportPropertiesConfig"].model_construct(),
        boussinesq_config=cfgs["BoussinesqConfig"].model_construct(),
    )


class _StubAgent:
    """Async stand-in for a pydantic-ai Agent (no network)."""

    def __init__(self, output: Any, on_run: Any = None) -> None:
        self._output = output
        self._on_run = on_run
        self.calls: list[Any] = []

    async def run(self, prompt: str, message_history: Any = None) -> SimpleNamespace:
        # Snapshot history (send_message mutates the same list in place afterwards).
        snap = list(message_history) if message_history is not None else None
        self.calls.append((prompt, snap))
        if self._on_run is not None:
            self._on_run()
        return SimpleNamespace(output=self._output, all_messages=lambda: ["MSG"])


def test_send_message_fills_forms_autoselects_and_logs():
    solver = _solver()
    server = _server()
    entries = build_forms(solver)
    prebuilt = _prebuilt_case_spec(solver)

    send = build_agent_panel(
        server,
        entries,
        solver,
        agent_factory=lambda *, solver, model_name: _StubAgent(prebuilt),
    )
    asyncio.run(send("buoyant hot-room, laminar"))

    # Transcript has the user turn + an assistant reply.
    roles = [m["role"] for m in server.state.chat_log]
    assert roles == ["user", "assistant"]
    assert "Filled" in server.state.chat_log[-1]["content"]

    # Filled configs' state written; Boussinesq auto-selected.
    tp_key = next(
        e.state_key for e in entries if e.config_name == "transport_properties_config"
    )
    assert isinstance(server.state[tp_key], dict)
    assert server.state["sel_boussinesq"] is True
    assert server.state.ai_busy is False
    assert server.state.chat_input == ""


def test_multi_turn_threads_message_history():
    solver = _solver()
    server = _server()
    entries = build_forms(solver)
    stub = _StubAgent(_prebuilt_case_spec(solver))

    send = build_agent_panel(
        server, entries, solver, agent_factory=lambda *, solver, model_name: stub
    )
    asyncio.run(send("first"))
    asyncio.run(send("second"))

    # First call gets empty history; second call gets the history from the first.
    assert stub.calls[0][1] == []
    assert stub.calls[1][1] == ["MSG"]
    assert [m["role"] for m in server.state.chat_log] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_empty_message_is_ignored():
    solver = _solver()
    server = _server()
    send = build_agent_panel(
        server,
        build_forms(solver),
        solver,
        agent_factory=lambda *, solver, model_name: _StubAgent(
            _prebuilt_case_spec(solver)
        ),
    )
    asyncio.run(send("   "))
    assert server.state.chat_log == []


def test_degrades_when_agent_unavailable():
    solver = _solver()
    server = _server()

    def _raise(**_kw: Any) -> Any:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    send = build_agent_panel(server, build_forms(solver), solver, agent_factory=_raise)
    asyncio.run(send("hello"))

    assert server.state.chat_log[-1]["role"] == "assistant"
    assert "AI unavailable" in server.state.chat_log[-1]["content"]
    assert server.state.ai_busy is False


class _StubGeoAgent:
    """Async stand-in for the geometry agent (returns fixed assignments)."""

    def __init__(self, output: Any) -> None:
        self._output = output
        self.calls: list[str] = []

    async def run(self, prompt: str) -> SimpleNamespace:
        self.calls.append(prompt)
        return SimpleNamespace(output=self._output)


def test_send_message_also_fills_geometry_roles():
    from neofoam.ui.geometry import PatchRole
    from neofoam.ui.geometry_agent import GeometryAssignments, RoleAssignment

    solver = _solver()
    server = _server()
    entries = build_forms(solver)
    assignments = GeometryAssignments(
        assignments=[
            RoleAssignment(patch="tubes", role=PatchRole.wall, refinement=(3, 4))
        ]
    )

    send = build_agent_panel(
        server,
        entries,
        solver,
        agent_factory=lambda *, solver, model_name: _StubAgent(
            _prebuilt_case_spec(solver)
        ),
        geometry_agent_factory=lambda: _StubGeoAgent(assignments),
    )
    # The scan populates patches after the panel is built; then a message fills them.
    server.state.geometry_patches = [
        {
            "name": "tubes",
            "role": "wall",
            "is_snappy": True,
            "box_faces": None,
            "refinement_str": "1 2",
        }
    ]
    asyncio.run(send("the tubes are heated walls, refine them"))

    row = server.state.geometry_patches[0]
    assert row["refinement"] == [3, 4]
    assert any("Mesh roles set" in m["content"] for m in server.state.chat_log)


def test_geometry_fill_skipped_when_no_patches():
    solver = _solver()
    server = _server()
    # geometry_patches defaults to [] → the geometry agent is never built/run.
    stub_geo = _StubGeoAgent(None)

    send = build_agent_panel(
        server,
        build_forms(solver),
        solver,
        agent_factory=lambda *, solver, model_name: _StubAgent(
            _prebuilt_case_spec(solver)
        ),
        geometry_agent_factory=lambda: stub_geo,
    )
    asyncio.run(send("laminar cavity"))
    assert stub_geo.calls == []


def test_busy_true_during_run():
    solver = _solver()
    server = _server()
    seen = {}
    stub = _StubAgent(
        _prebuilt_case_spec(solver),
        on_run=lambda: seen.__setitem__("busy", server.state.ai_busy),
    )

    send = build_agent_panel(
        server,
        build_forms(solver),
        solver,
        agent_factory=lambda *, solver, model_name: stub,
    )
    asyncio.run(send("go"))
    assert seen["busy"] is True
    assert server.state.ai_busy is False
