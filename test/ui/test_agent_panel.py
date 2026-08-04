# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""AI chat controller tests with a network-free stub agent (no browser, no network)."""

from __future__ import annotations

import asyncio
from pathlib import Path
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

#: A real on-disk case for the ``load_case`` tool (the fixture test/agent uses).
SOURCE_CASE = (
    Path(__file__).resolve().parents[1] / "solver" / "incompressibleFluid" / "val_pitzDaily"
)


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
        agent_factory=lambda **_kw: _StubAgent(prebuilt),
    )
    asyncio.run(send("buoyant hot-room, laminar"))

    # Transcript has the user turn + an assistant reply.
    roles = [m["role"] for m in server.state.chat_log]
    assert roles == ["user", "assistant"]
    assert "Filled" in server.state.chat_log[-1]["content"]

    # Filled configs' state written; Boussinesq auto-selected.
    tp_key = next(e.state_key for e in entries if e.config_name == "transport_properties_config")
    assert isinstance(server.state[tp_key], dict)
    assert server.state["sel_boussinesq"] is True
    assert server.state.ai_busy is False
    assert server.state.chat_input == ""


def test_multi_turn_threads_message_history():
    solver = _solver()
    server = _server()
    entries = build_forms(solver)
    stub = _StubAgent(_prebuilt_case_spec(solver))

    send = build_agent_panel(server, entries, solver, agent_factory=lambda **_kw: stub)
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
        agent_factory=lambda **_kw: _StubAgent(_prebuilt_case_spec(solver)),
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
    from neofoam.ui.geometry import PatchRole  # noqa: PLC0415
    from neofoam.ui.geometry_agent import GeometryAssignments, RoleAssignment  # noqa: PLC0415

    solver = _solver()
    server = _server()
    entries = build_forms(solver)
    assignments = GeometryAssignments(
        assignments=[RoleAssignment(patch="tubes", role=PatchRole.wall, refinement=(3, 4))]
    )

    send = build_agent_panel(
        server,
        entries,
        solver,
        agent_factory=lambda **_kw: _StubAgent(_prebuilt_case_spec(solver)),
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
        agent_factory=lambda **_kw: _StubAgent(_prebuilt_case_spec(solver)),
        geometry_agent_factory=lambda: stub_geo,
    )
    asyncio.run(send("laminar cavity"))
    assert stub_geo.calls == []


def test_chat_handler_owns_prompt_for_its_step():
    solver = _solver()
    server = _server()
    calls: list[str] = []

    async def handler(prompt: str) -> str:
        calls.append(prompt)
        return "handled: " + prompt

    send = build_agent_panel(
        server,
        build_forms(solver),
        solver,
        agent_factory=lambda **_kw: _StubAgent(_prebuilt_case_spec(solver)),
    )
    server.controller.register_chat_handler("cad", handler)

    # Off the handler's step → the physics fill runs; the handler is untouched.
    server.state.current_step = "models"
    asyncio.run(send("fill physics"))
    assert calls == []

    # On the handler's step → it owns the prompt and its reply is logged.
    server.state.current_step = "cad"
    asyncio.run(send("a tube bank"))
    assert calls == ["a tube bank"]
    assert server.state.chat_log[-1]["content"] == "handled: a tube bank"
    assert server.state.ai_busy is False


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
        agent_factory=lambda **_kw: stub,
    )
    asyncio.run(send("go"))
    assert seen["busy"] is True
    assert server.state.ai_busy is False


def _empty_case_spec(solver: Any) -> Any:
    """A CaseSpec with every field null — an agent that only called a tool."""
    return build_case_output_model(solver=solver)()


def _tp_key(entries: list[Any]) -> str:
    return next(e.state_key for e in entries if e.config_name == "transport_properties_config")


def _loading_factory(output: Any, case_dir: Any) -> tuple[Any, list[str]]:
    """Stub factory whose agent calls the injected ``load_case`` tool during ``run``."""
    replies: list[str] = []

    def factory(**kw: Any) -> Any:
        (load_case,) = kw["tools"]
        return _StubAgent(output, on_run=lambda: replies.append(load_case(str(case_dir))))

    return factory, replies


def test_load_case_tool_reads_the_case_into_the_forms():
    solver = _solver()
    server = _server()
    entries = build_forms(solver)
    factory, replies = _loading_factory(_empty_case_spec(solver), SOURCE_CASE)

    send = build_agent_panel(server, entries, solver, agent_factory=factory)
    asyncio.run(send(f"open the case at {SOURCE_CASE}"))

    # The disk values landed in the forms — the stub agent itself produced nothing.
    assert server.state[_tp_key(entries)] == {"transportModel": "Newtonian", "nu": 1e-05}
    assert "TransportPropertiesConfig" in replies[0]
    assert "**Loaded**" in server.state.chat_log[-1]["content"]


def test_loaded_case_is_auto_saved_to_the_target_dir(tmp_path):
    solver = _solver()
    server = _server()
    server.state.target_dir = str(tmp_path)
    entries = build_forms(solver)
    factory, _ = _loading_factory(_empty_case_spec(solver), SOURCE_CASE)

    send = build_agent_panel(server, entries, solver, agent_factory=factory)
    asyncio.run(send(f"open the case at {SOURCE_CASE}"))

    # The auto-save covers what the tool loaded, not just the agent's own output.
    assert (tmp_path / "constant" / "transportProperties").is_file()
    assert "transportProperties" in server.state.chat_log[-1]["content"]


def test_agent_output_overrides_the_loaded_case():
    solver = _solver()
    server = _server()
    entries = build_forms(solver)
    case_spec_cls = build_case_output_model(solver=solver)
    refined = case_spec_cls.model_construct(
        transport_properties_config=configurations(solver)[
            "TransportPropertiesConfig"
        ].model_construct(transportModel="CrossPowerLaw")
    )
    factory, _ = _loading_factory(refined, SOURCE_CASE)

    send = build_agent_panel(server, entries, solver, agent_factory=factory)
    asyncio.run(send(f"open {SOURCE_CASE} and switch to CrossPowerLaw"))

    # Same entry filled twice → the agent's refinement is applied last and wins.
    assert server.state[_tp_key(entries)] == {"transportModel": "CrossPowerLaw"}


def test_load_case_tool_reports_a_missing_directory(tmp_path):
    solver = _solver()
    server = _server()
    entries = build_forms(solver)
    missing = tmp_path / "nope"
    factory, replies = _loading_factory(_empty_case_spec(solver), missing)

    send = build_agent_panel(server, entries, solver, agent_factory=factory)
    asyncio.run(send(f"open {missing}"))

    assert replies == [f"No case directory at {missing}."]
    assert server.state[_tp_key(entries)] is None  # forms untouched (never seeded here)
    assert server.state.ai_busy is False


def test_loaded_case_is_not_reapplied_on_the_next_turn():
    solver = _solver()
    server = _server()
    entries = build_forms(solver)
    tools: list[Any] = []

    def factory(**kw: Any) -> Any:
        tools[:] = kw["tools"]
        # Only the first turn loads; the second is a plain no-op reply.
        return _StubAgent(
            _empty_case_spec(solver),
            on_run=lambda: tools[0](str(SOURCE_CASE)) if not server.state.chat_log[2:] else None,
        )

    send = build_agent_panel(server, entries, solver, agent_factory=factory)
    asyncio.run(send("open the case"))
    server.state[_tp_key(entries)] = {"transportModel": "edited by hand"}

    asyncio.run(send("thanks"))
    assert server.state[_tp_key(entries)] == {"transportModel": "edited by hand"}
