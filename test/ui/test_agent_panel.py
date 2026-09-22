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

from neofoam.agent.case_fill import build_case_output_model, case_spec_to_configs  # noqa: E402
from neofoam.framework.solver.configurations import configurations  # noqa: E402
from neofoam.ui.agent_panel import _summary, build_agent_panel  # noqa: E402
from neofoam.ui.forms import build_forms  # noqa: E402

#: A real on-disk case for the ``load_case`` tool (the fixture test/agent uses).
SOURCE_CASE = (
    Path(__file__).resolve().parents[1] / "solver" / "incompressibleFluid" / "val_pitzDaily"
)

#: A checked-in case that carries its fields in ``0/`` (read-only); ``SOURCE_CASE`` has
#: them in ``0.orig/`` only.
ZERO_DIR_CASE = Path(__file__).resolve().parents[1] / "setup_pimple"

#: A case whose ``p`` block holds a ``preconditioner`` beside ``solver GAMG`` (read-only).
STALE_COMPANION_CASE = Path(__file__).resolve().parent / "cases" / "stale_companion"

#: A case whose only file, ``transportProperties``, holds a ``nu`` that is no number.
INVALID_TRANSPORT_CASE = Path(__file__).resolve().parent / "cases" / "invalid_transport"


@pytest.fixture
def server(request: pytest.FixtureRequest) -> Any:
    """A bare trame server named after the requesting test (trame keeps one per name)."""
    srv = get_server(f"neofoam_ui_agent_{request.node.name}")
    srv.state.target_dir = ""  # no auto-save in tests
    return srv


@pytest.fixture
def entries(solver: Any) -> list[Any]:
    """The form entries the panel under test fills."""
    return build_forms(solver)


#: A case whose only file holds a stray ``}``: OpenFOAM exits the process that parses it.
UNPARSABLE_CASE = Path(__file__).resolve().parent / "cases" / "unparsable_dict"


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


def test_send_message_fills_forms_autoselects_and_logs(solver, server, entries):
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


def test_multi_turn_threads_message_history(solver, server, entries):
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


def test_empty_message_is_ignored(solver, server, entries):
    send = build_agent_panel(
        server,
        entries,
        solver,
        agent_factory=lambda **_kw: _StubAgent(_prebuilt_case_spec(solver)),
    )
    asyncio.run(send("   "))
    assert server.state.chat_log == []


def test_degrades_when_agent_unavailable(solver, server, entries):
    def _raise(**_kw: Any) -> Any:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    send = build_agent_panel(server, entries, solver, agent_factory=_raise)
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


def test_send_message_also_fills_geometry_roles(solver, server, entries):
    from neofoam.ui.geometry import PatchRole  # noqa: PLC0415
    from neofoam.ui.geometry_agent import GeometryAssignments, RoleAssignment  # noqa: PLC0415

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
    assert row["refinement_str"] == "3 4"
    assert any("Mesh roles set" in m["content"] for m in server.state.chat_log)


def test_geometry_fill_skipped_when_no_patches(solver, server, entries):
    # geometry_patches defaults to [] → the geometry agent is never built/run.
    stub_geo = _StubGeoAgent(None)

    send = build_agent_panel(
        server,
        entries,
        solver,
        agent_factory=lambda **_kw: _StubAgent(_prebuilt_case_spec(solver)),
        geometry_agent_factory=lambda: stub_geo,
    )
    asyncio.run(send("laminar cavity"))
    assert stub_geo.calls == []


def test_chat_handler_owns_prompt_for_its_step(solver, server, entries):
    calls: list[str] = []

    async def handler(prompt: str) -> str:
        calls.append(prompt)
        return "handled: " + prompt

    send = build_agent_panel(
        server,
        entries,
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


def test_busy_true_during_run(solver, server, entries):
    seen = {}
    stub = _StubAgent(
        _prebuilt_case_spec(solver),
        on_run=lambda: seen.__setitem__("busy", server.state.ai_busy),
    )

    send = build_agent_panel(
        server,
        entries,
        solver,
        agent_factory=lambda **_kw: stub,
    )
    asyncio.run(send("go"))
    assert seen["busy"] is True
    assert server.state.ai_busy is False


class _BlockingAgent:
    """Stub agent whose first run parks until released (no network)."""

    def __init__(self, output: Any) -> None:
        self._output = output
        self.calls: list[Any] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, prompt: str, message_history: Any = None) -> SimpleNamespace:
        snap = list(message_history) if message_history is not None else None
        self.calls.append((prompt, snap))
        if len(self.calls) == 1:
            self.started.set()
            await self.release.wait()
        return SimpleNamespace(output=self._output, all_messages=lambda: ["MSG"])


def test_message_sent_while_busy_is_dropped(solver, server, entries):
    # Two overlapping turns both start from an empty message_history and both
    # rewrite it in place, so the second silently discards the first turn.
    stub = _BlockingAgent(_prebuilt_case_spec(solver))
    send = build_agent_panel(server, entries, solver, agent_factory=lambda **_kw: stub)

    async def drive() -> None:
        first = asyncio.create_task(send("first"))
        await stub.started.wait()
        await send("second")  # overlaps the running turn
        stub.release.set()
        await first

    asyncio.run(drive())

    assert [prompt for prompt, _ in stub.calls] == ["first"]
    assert [m["role"] for m in server.state.chat_log] == ["user", "assistant"]
    assert server.state.ai_busy is False


def _must_not_write(*_args: Any, **_kw: Any) -> list[str]:
    raise AssertionError("_summary wrote to disk")


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


def test_load_case_tool_reads_the_case_into_the_forms(solver, server, entries):
    factory, replies = _loading_factory(_empty_case_spec(solver), SOURCE_CASE)

    send = build_agent_panel(server, entries, solver, agent_factory=factory)
    asyncio.run(send(f"open the case at {SOURCE_CASE}"))

    # The disk values landed in the forms — the stub agent itself produced nothing.
    assert server.state[_tp_key(entries)] == {"transportModel": "Newtonian", "nu": 1e-05}
    assert "TransportPropertiesConfig" in replies[0]
    assert "**Loaded**" in server.state.chat_log[-1]["content"]


def test_loaded_case_is_auto_saved_to_the_target_dir(tmp_path, solver, server, entries):
    server.state.target_dir = str(tmp_path)
    factory, _ = _loading_factory(_empty_case_spec(solver), SOURCE_CASE)

    send = build_agent_panel(server, entries, solver, agent_factory=factory)
    asyncio.run(send(f"open the case at {SOURCE_CASE}"))

    # The auto-save covers what the tool loaded, not just the agent's own output.
    assert (tmp_path / "constant" / "transportProperties").is_file()
    assert "transportProperties" in server.state.chat_log[-1]["content"]


def test_agent_output_overrides_the_loaded_case(solver, server, entries):
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


def _p_block(server: Any, entries: list[Any]) -> dict[str, Any]:
    key = next(e.state_key for e in entries if e.config_name == "pimple_fv_solution")
    return server.state[key]["solvers"]["p"]


def test_agent_fill_drops_the_companion_its_solver_does_not_take(solver, server, entries):
    defaults = next(e.defaults for e in entries if e.config_name == "pimple_fv_solution")
    block = {**defaults["solvers"]["p"], "solver": "GAMG", "relTol": 0.01}
    filled = build_case_output_model(solver=solver).model_validate(
        {"pimple_fv_solution": {**defaults, "solvers": {**defaults["solvers"], "p": block}}}
    )

    send = build_agent_panel(
        server, entries, solver, agent_factory=lambda **_kw: _StubAgent(filled)
    )
    asyncio.run(send("solve p with GAMG, relTol 0.01"))

    assert _p_block(server, entries) == {"solver": "GAMG", "tolerance": 1e-06, "relTol": 0.01}


def test_loaded_case_keeps_a_companion_its_solver_does_not_take(solver, server, entries):
    factory, _ = _loading_factory(_empty_case_spec(solver), STALE_COMPANION_CASE)

    send = build_agent_panel(server, entries, solver, agent_factory=factory)
    asyncio.run(send(f"open the case at {STALE_COMPANION_CASE}"))

    # A file the user wrote is shown as it is, never silently rewritten.
    assert _p_block(server, entries)["preconditioner"] == "DIC"


def test_load_case_tool_reports_a_missing_directory(tmp_path, solver, server, entries):
    missing = tmp_path / "nope"
    factory, replies = _loading_factory(_empty_case_spec(solver), missing)

    send = build_agent_panel(server, entries, solver, agent_factory=factory)
    asyncio.run(send(f"open {missing}"))

    assert replies == [f"No case directory at {missing}."]
    assert server.state[_tp_key(entries)] is None  # forms untouched (never seeded here)
    assert server.state.ai_busy is False


def test_loaded_case_is_not_reapplied_on_the_next_turn(solver, server, entries):
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


def _must_not_build_an_agent(**kw: Any) -> Any:
    raise AssertionError("the Load case button must not need the agent (no API key)")


def test_load_target_case_fills_the_forms_without_an_agent(solver, server, entries):
    build_agent_panel(server, entries, solver, agent_factory=_must_not_build_an_agent)
    server.state.target_dir = str(SOURCE_CASE)

    server.controller.load_target_case()

    assert server.state[_tp_key(entries)] == {"transportModel": "Newtonian", "nu": 1e-05}
    assert f"**Loaded** `{SOURCE_CASE}`" in server.state.chat_log[-1]["content"]


def test_load_target_case_fills_the_field_forms_from_the_zero_directory(solver, server, entries):
    build_agent_panel(server, entries, solver)
    server.state.target_dir = str(ZERO_DIR_CASE)

    server.controller.load_target_case()

    forms = {e.key: server.state[e.state_key] for e in entries if e.cls_name == "pFieldConfig"}
    assert forms == {
        "field_in:pFieldConfig": {"dimensions": "[0 2 -2 0 0 0 0]", "internalField": "uniform 0"},
        "field_bc:pFieldConfig": {
            "boundaryField": {
                "movingWall": {"type": "zeroGradient"},
                "fixedWalls": {"type": "zeroGradient"},
                "frontAndBack": {"type": "empty"},
            }
        },
    }
    assert "0.orig" not in server.state.chat_log[-1]["content"]


def test_load_target_case_reports_fields_kept_in_zero_orig_only(solver, server, entries):
    build_agent_panel(server, entries, solver)
    server.state.target_dir = str(SOURCE_CASE)

    server.controller.load_target_case()

    # A load replaces: a form the case does not fill holds its defaults, not case data.
    p_entries = [e for e in entries if e.cls_name == "pFieldConfig"]
    forms = {e.key: server.state[e.state_key] for e in p_entries}
    assert forms == {e.key: dict(e.defaults) for e in p_entries}
    assert "No `0/` directory: the fields in `0.orig/` were not loaded." in (
        server.state.chat_log[-1]["content"].split("\n\n")
    )


def test_load_target_case_opens_the_assistant_drawer_for_its_report(solver, server, entries):
    # The report lands in the chat, which is folded by default on a phone.
    build_agent_panel(server, entries, solver)
    server.state.update({"target_dir": str(SOURCE_CASE), "ai_panel": False})

    server.controller.load_target_case()

    assert server.state.ai_panel is True
    assert server.state.ai_panel_mobile is True


@pytest.mark.parametrize(
    ("target", "reported"),
    [
        ("relative/case", "The target directory must be an absolute path, got 'relative/case'."),
        ("/no/such/neofoam/case", "No case directory at /no/such/neofoam/case."),
        (
            str(INVALID_TRANSPORT_CASE),
            f"No case files found in {INVALID_TRANSPORT_CASE}. Present but invalid:"
            " constant/transportProperties (could not convert string to float: 'notANumber').",
        ),
        (
            str(UNPARSABLE_CASE),
            f"Could not read {UNPARSABLE_CASE}: Unexpected '}}' while reading dictionary entry"
            f" ({UNPARSABLE_CASE}/constant/transportProperties at line 14).",
        ),
    ],
)
def test_load_target_case_reports_a_directory_it_cannot_load(
    solver, target, reported, server, entries
):
    build_agent_panel(server, entries, solver)
    server.state.target_dir = target

    server.controller.load_target_case()

    assert server.state.chat_log == [{"role": "assistant", "content": reported}]
    assert server.state[_tp_key(entries)] is None  # forms untouched (never seeded here)


def test_assistant_report_is_kept_as_escaped_html_for_the_chat(tmp_path, solver, server, entries):
    # The bubble binds chat_html with v-html; a path is untrusted text.
    build_agent_panel(server, entries, solver)
    server.state.target_dir = str(tmp_path / "<img src=x onerror=alert(1)>")

    server.controller.load_target_case()

    assert server.state.chat_html == [
        f"No case directory at {tmp_path}/&lt;img src=x onerror=alert(1)&gt;."
    ]


def test_user_message_has_no_html(solver, server, entries):
    async def handler(prompt: str) -> str:
        return "**done**"

    build_agent_panel(server, entries, solver)
    server.controller.register_chat_handler("models", handler)
    server.state.current_step = "models"

    asyncio.run(server.controller.send_message("<b>hi</b>"))

    assert server.state.chat_html == ["", "<strong>done</strong>"]


def test_load_target_case_reports_an_empty_directory(tmp_path, solver, server, entries):
    build_agent_panel(server, entries, solver)
    server.state.target_dir = str(tmp_path)

    server.controller.load_target_case()

    assert server.state.chat_log[-1]["content"] == f"No case files found in {tmp_path}."


def test_load_target_case_is_dropped_while_the_assistant_runs(solver, server, entries):
    # A chat turn applies its own load when it ends; a second writer would race it.
    build_agent_panel(server, entries, solver)
    server.state.update({"target_dir": str(SOURCE_CASE), "ai_busy": True})

    server.controller.load_target_case()

    assert server.state[_tp_key(entries)] is None
    assert server.state.chat_log == []


def test_summary_is_pure_string_building(tmp_path, monkeypatch, solver):
    configs = case_spec_to_configs(_prebuilt_case_spec(solver))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("neofoam.ui.agent_panel.write_configs", _must_not_write)

    text = _summary(configs, {"boussinesq"}, None, ["**Wrote to** `/case`:", "- g"])

    assert text == (
        "**Filled:** BoussinesqConfig, TransportPropertiesConfig\n\n"
        "**Selected models:** boussinesq\n\n"
        "**Wrote to** `/case`:\n\n"
        "- g\n\n"
        "Review the forms; click **Save case** when ready."
    )
    assert list(tmp_path.iterdir()) == []


def test_agent_output_is_auto_saved_before_the_reply(tmp_path, solver, server, entries):
    server.state.target_dir = str(tmp_path)
    refined = build_case_output_model(solver=solver).model_construct(
        transport_properties_config=configurations(solver)[
            "TransportPropertiesConfig"
        ].model_construct(transportModel="Newtonian", nu=1e-05)
    )

    send = build_agent_panel(
        server, entries, solver, agent_factory=lambda **_kw: _StubAgent(refined)
    )
    asyncio.run(send("water, laminar"))

    assert (tmp_path / "constant" / "transportProperties").is_file()
    assert server.state.chat_log[-1]["content"] == (
        "**Filled:** TransportPropertiesConfig\n\n"
        f"**Wrote to** `{tmp_path}`:\n\n"
        "- transportProperties\n\n"
        "Review the forms; click **Save case** when ready."
    )


@pytest.mark.parametrize("target", ["relative_case", "   "])
def test_auto_save_never_writes_to_the_launch_dir(
    target, tmp_path, monkeypatch, solver, server, entries
):
    # The auto-save resolves its target like every other writer, so a relative or
    # blank field cannot drop the generated configs where the server was started.
    server.state.target_dir = target
    monkeypatch.chdir(tmp_path)
    refined = build_case_output_model(solver=solver).model_construct(
        transport_properties_config=configurations(solver)[
            "TransportPropertiesConfig"
        ].model_construct(transportModel="Newtonian", nu=1e-05)
    )

    send = build_agent_panel(
        server, entries, solver, agent_factory=lambda **_kw: _StubAgent(refined)
    )
    asyncio.run(send("water, laminar"))

    assert list(tmp_path.iterdir()) == []
    assert "**Wrote to**" not in server.state.chat_log[-1]["content"]
