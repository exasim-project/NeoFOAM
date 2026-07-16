# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tool-graph resolution + DAG chaining (pure-Python, no mesh executed).

Uses in-test stub tools whose ``@build`` emits a single recording :class:`InitStep`,
so DAG ordering, ``_prev_mesh`` threading, and the terminal ``mesh`` sink alias can be
verified without pybFoam. Execution order comes from each entry's ``depends_on``, never
its list position.
"""

from pathlib import Path
from typing import Any

import pytest

from neofoam.framework.initialization import InitStep, lazy
from neofoam.framework.initialization.execution import (
    InitializationGraphError,
    execute_initialization,
)
from neofoam.framework.tools import (
    PreprocessConfig,
    Tool,
    resolve_tools,
    tool_graph_steps,
)

CASE = Path(__file__).parents[2] / "solver" / "incompressibleFluid" / "preprocess_case"


def _stub_tool(
    name: str,
    seen: dict[str, Any],
    produced: dict[str, Any],
    *,
    category: str = "resource",
) -> Any:
    """A pybFoam-free tool whose build emits one recording step.

    The step records the ``_prev_mesh`` it saw and returns a unique sentinel,
    so threading and ordering are observable.
    """
    t = Tool(name)
    # Every step advances the mesh; it returns an opaque sentinel so threading
    # and ordering are observable.
    out: Any = object()
    produced[name] = out

    @t.build
    def _b(cfg: Any) -> list[InitStep]:
        def init(ctx: dict[str, Any]) -> Any:
            seen[name] = ctx.get("_prev_mesh")
            return out

        return [
            InitStep(name=f"preprocess.{name}", initializer=init, category=category)
        ]

    return t


def _recording_tool(name: str, order: list[str]) -> Any:
    """A tool whose build step appends its name to ``order`` when executed."""
    t = Tool(name)
    out: Any = object()

    @t.build
    def _b(cfg: Any) -> list[InitStep]:
        def init(ctx: dict[str, Any]) -> Any:
            order.append(name)
            return out

        return [InitStep(name=f"preprocess.{name}", initializer=init)]

    return t


def test_resolve_tools_resolves_registered_tool() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    rts = resolve_tools([a], PreprocessConfig(tools=[{"tool": "a"}]))
    assert [rt.name for rt in rts] == ["preprocess.a"]


def test_resolve_tools_unknown_tool_raises() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    b = _stub_tool("b", seen, produced)
    with pytest.raises(ValueError, match="a"):
        resolve_tools([], PreprocessConfig(tools=[{"tool": "a"}]))
    with pytest.raises(ValueError, match="a"):
        resolve_tools([b], PreprocessConfig(tools=[{"tool": "a"}]))


def test_resolve_tools_carries_depends_on() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    b = _stub_tool("b", seen, produced)
    rts = resolve_tools(
        [a, b],
        PreprocessConfig(tools=[{"tool": "a"}, {"tool": "b", "depends_on": ["a"]}]),
    )
    assert [rt.name for rt in rts] == ["preprocess.a", "preprocess.b"]
    assert rts[0].depends_on == []
    assert rts[1].depends_on == ["a"]


def test_resolve_tools_empty_is_empty() -> None:
    assert resolve_tools([], PreprocessConfig(tools=[])) == []


def test_dependency_becomes_initstep_depends_on() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    block = _stub_tool("block", seen, produced)
    snappy = _stub_tool("snappy", seen, produced)
    steps = tool_graph_steps(
        [
            block.instantiate({"tool": "block"}),
            snappy.instantiate({"tool": "snappy", "depends_on": ["block"]}),
        ]
    )
    by_name = {s.name: s for s in steps}
    assert by_name["preprocess.block"].depends_on == ["_foam_time"]
    assert by_name["preprocess.snappy"].depends_on == [
        "_foam_time",
        "preprocess.block",
    ]


def test_entry_without_depends_on_is_root() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    block = _stub_tool("block", seen, produced)
    steps = tool_graph_steps([block.instantiate({"tool": "block"})])
    assert [s.name for s in steps] == ["preprocess.block", "mesh"]
    assert steps[0].depends_on == ["_foam_time"]
    assert steps[-1].depends_on == ["preprocess.block"]
    assert steps[-1].replaces == ["mesh"]


def test_scrambled_file_order_yields_dag_order() -> None:
    order: list[str] = []
    a = _recording_tool("a", order)
    b = _recording_tool("b", order)
    c = _recording_tool("c", order)
    # scrambled file order: c (after b), b (after a), a (root)
    rts = resolve_tools(
        [a, b, c],
        PreprocessConfig(
            tools=[
                {"tool": "c", "depends_on": ["b"]},
                {"tool": "b", "depends_on": ["a"]},
                {"tool": "a"},
            ]
        ),
    )
    steps = [lazy("_foam_time", lambda _ctx: "T0")]
    steps.extend(tool_graph_steps(rts))
    execute_initialization(steps)
    assert order == ["a", "b", "c"]  # DAG order, not file order


def test_prev_mesh_sourced_from_declared_dependency() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    b = _stub_tool("b", seen, produced)
    c = _stub_tool("c", seen, produced)
    rts = resolve_tools(
        [a, b, c],
        PreprocessConfig(
            tools=[
                {"tool": "c", "depends_on": ["b"]},
                {"tool": "b", "depends_on": ["a"]},
                {"tool": "a"},
            ]
        ),
    )
    steps = [lazy("_foam_time", lambda _ctx: "T0")]
    steps.extend(tool_graph_steps(rts))
    ctx = execute_initialization(steps)
    assert seen["a"] is None  # root threads no _prev_mesh
    assert seen["b"] is produced["a"]  # b reads its declared dep a's output
    assert seen["c"] is produced["b"]  # c reads its declared dep b's output
    assert ctx.mesh is produced["c"]  # sink (c) published as the mesh


def test_absent_dependency_raises_graph_error() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    c = _stub_tool("c", seen, produced)
    steps = [lazy("_foam_time", lambda _ctx: "T0")]
    steps.extend(
        tool_graph_steps([c.instantiate({"tool": "c", "depends_on": ["nope"]})])
    )
    with pytest.raises(InitializationGraphError):
        execute_initialization(steps)


def test_dependency_cycle_raises_graph_error() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    b = _stub_tool("b", seen, produced)
    steps = [lazy("_foam_time", lambda _ctx: "T0")]
    steps.extend(
        tool_graph_steps(
            [
                a.instantiate({"tool": "a", "depends_on": ["b"]}),
                b.instantiate({"tool": "b", "depends_on": ["a"]}),
            ]
        )
    )
    with pytest.raises(InitializationGraphError):
        execute_initialization(steps)


def test_multiple_sinks_raise_valueerror() -> None:
    # Two tools, neither depended on by the other → two graph sinks. There can be
    # only one tool that owns the published mesh, so wiring must reject this clearly.
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    b = _stub_tool("b", seen, produced)
    with pytest.raises(ValueError, match="exactly one sink"):
        tool_graph_steps([a.instantiate({"tool": "a"}), b.instantiate({"tool": "b"})])


def test_multiple_depends_on_raises_valueerror() -> None:
    # Mesh threading is single-predecessor by design: a tool declaring >1 dependency
    # has no unambiguous _prev_mesh, so wiring rejects it with a clear error naming
    # the tool and its deps rather than deferring to a bare KeyError in @build.
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    b = _stub_tool("b", seen, produced)
    c = _stub_tool("c", seen, produced)
    with pytest.raises(ValueError, match="single-predecessor"):
        tool_graph_steps(
            [
                a.instantiate({"tool": "a"}),
                b.instantiate({"tool": "b"}),
                c.instantiate({"tool": "c", "depends_on": ["a", "b"]}),
            ]
        )


def test_empty_build_raises_clear_error() -> None:
    # A tool whose @build returns no steps must raise a clear error naming it,
    # not a bare IndexError.
    t = Tool("noop")

    @t.build
    def _b(cfg: Any) -> list[InitStep]:
        return []

    with pytest.raises(ValueError, match="noop"):
        tool_graph_steps([t.instantiate({"tool": "noop"})])


def test_empty_graph_produces_no_steps() -> None:
    assert tool_graph_steps([]) == []


def test_untyped_tool_with_depends_on_orders() -> None:
    # Open-seam tool (untyped @build) resolves a raw mapping AND DAG-wires.
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    u = _stub_tool("u", seen, produced)
    rts = resolve_tools(
        [a, u],
        PreprocessConfig(
            tools=[
                {"tool": "u", "depends_on": ["a"], "extra": 1},
                {"tool": "a"},
            ]
        ),
    )
    assert rts[0].config == {"tool": "u", "depends_on": ["a"], "extra": 1}
    assert rts[0].depends_on == ["a"]
    steps = [lazy("_foam_time", lambda _ctx: "T0")]
    steps.extend(tool_graph_steps(rts))
    ctx = execute_initialization(steps)
    assert ctx.mesh is produced["u"]  # u is the sink


def test_graph_loads_open_entries_with_depends_on() -> None:
    # The schema is open: each entry is a raw mapping keyed on ``tool`` carrying an
    # optional ``depends_on`` envelope; the DAG (not list position) decides order.
    cfg = PreprocessConfig.load(case_dir=CASE)
    assert [entry["tool"] for entry in cfg.tools] == [
        "blockMesh",
        "snappyHexMesh",
        "checkMesh",
    ]
    by_tool = {entry["tool"]: entry for entry in cfg.tools}
    assert "depends_on" not in by_tool["blockMesh"]
    assert by_tool["snappyHexMesh"]["depends_on"] == ["blockMesh"]
    assert by_tool["checkMesh"]["depends_on"] == ["snappyHexMesh"]


def test_absent_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PreprocessConfig.load(case_dir=tmp_path)


def _consuming_tool(name: str, seen: dict[str, Any]) -> Any:
    """A consumes_mesh tool (snappy-like): its build reads ``_prev_mesh``."""
    t = Tool(name, consumes_mesh=True)
    out: Any = object()

    @t.build
    def _b(cfg: Any) -> list[InitStep]:
        def init(ctx: dict[str, Any]) -> Any:
            seen[name] = ctx["_prev_mesh"]
            return out

        return [InitStep(name=f"preprocess.{name}", initializer=init)]

    return t


def test_dependency_less_consumer_seeded_from_mesh_source() -> None:
    # A single-tool snappy-like slice resumes from a mesh already on disk: the
    # injected mesh_source provides _prev_mesh.
    seen: dict[str, Any] = {}
    disk_mesh = object()
    rt = _consuming_tool("snappyLike", seen).instantiate({"tool": "snappyLike"})

    steps = tool_graph_steps([rt], mesh_source=lambda ctx: disk_mesh)
    steps.append(lazy("_foam_time", lambda ctx: object()))
    ctx = execute_initialization(steps)
    assert seen["snappyLike"] is disk_mesh
    # The consumer is still the sink → published as the terminal mesh alias.
    assert ctx.mesh is not None


def test_dependency_less_consumer_without_mesh_source_raises() -> None:
    rt = _consuming_tool("snappyLike", {}).instantiate({"tool": "snappyLike"})
    with pytest.raises(ValueError, match="snappyLike.*no mesh source"):
        tool_graph_steps([rt])


def test_mesh_source_not_used_for_creators_or_chained_consumers() -> None:
    # A full blockMesh -> snappy chain must NOT read the disk mesh: snappy's
    # _prev_mesh is its declared dependency's output, exactly as before.
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    creator = _stub_tool("creatorTool", seen, produced)
    consumer = _consuming_tool("chainedConsumer", seen)
    runtimes = [
        creator.instantiate({"tool": "creatorTool"}),
        consumer.instantiate(
            {"tool": "chainedConsumer", "depends_on": ["creatorTool"]}
        ),
    ]
    disk_reads: list[bool] = []

    def source(ctx: dict[str, Any]) -> Any:
        disk_reads.append(True)
        return object()

    steps = tool_graph_steps(runtimes, mesh_source=source)
    steps.append(lazy("_foam_time", lambda ctx: object()))
    execute_initialization(steps)
    assert seen["chainedConsumer"] is produced["creatorTool"]
    assert disk_reads == []  # lazily declared, never needed → never read
