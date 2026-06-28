# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pipeline resolution + chaining (pure-Python, no mesh executed).

Uses in-test stub tools whose ``@build`` emits a single recording :class:`InitStep`,
so the linear chain, ``_prev_mesh`` threading, and terminal ``mesh`` alias can be
verified without pybFoam.
"""

from pathlib import Path
from typing import Any

import pytest

from neofoam.framework.initialization import InitStep, lazy
from neofoam.framework.initialization.execution import (
    MESH_STATS_CATEGORY,
    execute_initialization,
)
from neofoam.framework.tools import (
    PreprocessConfig,
    Tool,
    resolve_pipeline,
    tool_init_steps,
)

CASE = Path(__file__).parents[2] / "solver" / "incompressibleFluid" / "preprocess_case"
EMPTY_CASE = (
    Path(__file__).parents[2]
    / "solver"
    / "incompressibleFluid"
    / "preprocess_case_empty"
)


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
    # A stats-category step's result is routed onto Context.mesh_stats, which is
    # validated as a dict; mesh-producing steps return an opaque sentinel.
    out: Any = {} if category == MESH_STATS_CATEGORY else object()
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


def test_resolve_pipeline_resolves_registered_tool() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    rts = resolve_pipeline([a], PreprocessConfig(pipeline=[{"tool": "a"}]))
    assert [rt.name for rt in rts] == ["preprocess.a"]


def test_resolve_pipeline_unknown_tool_raises() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    b = _stub_tool("b", seen, produced)
    with pytest.raises(ValueError, match="a"):
        resolve_pipeline([], PreprocessConfig(pipeline=[{"tool": "a"}]))
    with pytest.raises(ValueError, match="a"):
        resolve_pipeline([b], PreprocessConfig(pipeline=[{"tool": "a"}]))


def test_resolve_pipeline_preserves_order() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    b = _stub_tool("b", seen, produced)
    rts = resolve_pipeline(
        [a, b], PreprocessConfig(pipeline=[{"tool": "a"}, {"tool": "b"}])
    )
    assert [rt.name for rt in rts] == ["preprocess.a", "preprocess.b"]


def test_resolve_pipeline_empty_is_empty() -> None:
    assert resolve_pipeline([], PreprocessConfig(pipeline=[])) == []


def test_tool_init_steps_follow_pipeline_order() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    b = _stub_tool("b", seen, produced)
    c = _stub_tool("c", seen, produced, category=MESH_STATS_CATEGORY)
    runtimes = [
        a.instantiate({"tool": "a"}),
        b.instantiate({"tool": "b"}),
        c.instantiate({"tool": "c"}),
    ]
    steps = tool_init_steps(runtimes)
    assert [s.name for s in steps] == [
        "preprocess.a",
        "preprocess.b",
        "preprocess.c",
        "mesh",
    ]
    assert [s.depends_on for s in steps] == [
        ["_foam_time"],
        ["preprocess.a"],
        ["preprocess.b"],
        ["preprocess.c"],
    ]
    assert steps[-1].replaces == ["mesh"]


def test_tool_init_steps_threads_prev_mesh() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    b = _stub_tool("b", seen, produced)
    c = _stub_tool("c", seen, produced, category=MESH_STATS_CATEGORY)
    runtimes = [
        a.instantiate({"tool": "a"}),
        b.instantiate({"tool": "b"}),
        c.instantiate({"tool": "c"}),
    ]
    steps = [lazy("_foam_time", lambda _ctx: "T0")]
    steps.extend(tool_init_steps(runtimes))
    ctx = execute_initialization(steps)

    assert seen["a"] is None  # first step sees no prior mesh
    assert seen["b"] is produced["a"]  # b refines a's result
    assert seen["c"] is produced["b"]  # stats step consumes the last producer
    # terminal alias publishes the producer (b), not the stats step (c)
    assert ctx.mesh is produced["b"]


def test_stats_only_pipeline_emits_no_mesh_alias() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    c = _stub_tool("c", seen, produced, category=MESH_STATS_CATEGORY)
    steps = tool_init_steps([c.instantiate({"tool": "c"})])
    assert [s.name for s in steps] == ["preprocess.c"]
    assert all(s.name != "mesh" for s in steps)


def test_empty_pipeline_produces_no_steps() -> None:
    assert tool_init_steps([]) == []


def test_untyped_tool_passes_raw_mapping_and_chains() -> None:
    seen: dict[str, Any] = {}
    produced: dict[str, Any] = {}
    a = _stub_tool("a", seen, produced)
    u = _stub_tool("u", seen, produced)
    rts = resolve_pipeline(
        [a, u],
        PreprocessConfig(pipeline=[{"tool": "a"}, {"tool": "u", "extra": 1}]),
    )
    assert rts[1].config == {"tool": "u", "extra": 1}  # raw mapping passthrough
    steps = tool_init_steps(rts)
    assert [s.name for s in steps] == ["preprocess.a", "preprocess.u", "mesh"]


def test_pipeline_loads_ordered_open_entries() -> None:
    # The schema is open: each entry is a raw mapping keyed on ``tool`` (resolution
    # turns it into a typed step config), so ordering is what matters.
    cfg = PreprocessConfig.load(case_dir=CASE)
    assert [entry["tool"] for entry in cfg.pipeline] == [
        "blockMesh",
        "snappyHexMesh",
        "checkMesh",
    ]


def test_absent_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PreprocessConfig.load(case_dir=tmp_path)


def test_empty_pipeline_is_empty_list() -> None:
    cfg = PreprocessConfig.load(case_dir=EMPTY_CASE)
    assert cfg.pipeline == []
