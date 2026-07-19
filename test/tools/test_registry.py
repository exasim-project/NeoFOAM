# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared tool registry: self-registration, OCP resolution, cross-tool DAG + guards.

Importing ``neofoam.tools`` self-registers the three built-in tools; resolution is
scoped to that set. The migration grep guards assert the renamed/removed surfaces are
gone from ``src/`` + ``test/`` (needles are string-concatenated so this file does not
flag itself).
"""

import types
from pathlib import Path

import pytest

from neofoam.framework.initialization import (
    InitStepExecutionError,
    lazy,
)
from neofoam.framework.initialization.execution import (
    execute_initialization,
)
from neofoam.framework.tools import (
    PreprocessConfig,
    Tool,
    resolve_tools,
    tool_graph_steps,
)
from neofoam.tools import (
    available_tools,
    block_mesh,
    check_mesh,
    register_tool,
    registry,
    snappy_hex_mesh,
)
from neofoam.tools.block_mesh import blockMeshTool
from neofoam.tools.check_mesh import checkMeshTool
from neofoam.tools.run import detect_tools
from neofoam.tools.snappy_hex_mesh import snappyHexMeshTool

CASE = Path(__file__).parents[1] / "solver" / "incompressibleFluid" / "preprocess_case"

SRC = Path(__file__).parents[2] / "src" / "neofoam"
TEST = Path(__file__).parents[2] / "test"


def _fake_mesh_bindings(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], object]:
    """Fake all three tool modules' bindings; return (recorded calls, block sentinel)."""
    calls: list[str] = []
    block = object()
    for mod in (block_mesh, snappy_hex_mesh):
        monkeypatch.setattr(
            mod,
            "pyf",
            types.SimpleNamespace(dictionary=types.SimpleNamespace(read=lambda f: f)),
        )
    monkeypatch.setattr(
        block_mesh,
        "generate_blockmesh",
        lambda *a, **k: (calls.append("block"), block)[1],
    )
    monkeypatch.setattr(
        snappy_hex_mesh,
        "generate_snappy_hex_mesh",
        lambda *a, **k: calls.append("snappy"),
    )
    monkeypatch.setattr(
        check_mesh,
        "checkMesh",
        lambda *a, **k: (calls.append("check"), {"passed": True, "total_errors": 0})[1],
    )
    return calls, block


# self-registration ------------------------------------------------------
def test_builtin_tools_self_registered() -> None:
    names = {t.name for t in available_tools()}
    assert {"blockMesh", "snappyHexMesh", "checkMesh"} <= names


def test_registered_tools_are_the_module_singletons() -> None:
    by_name = {t.name: t for t in available_tools()}
    assert by_name["blockMesh"] is blockMeshTool
    assert by_name["snappyHexMesh"] is snappyHexMeshTool
    assert by_name["checkMesh"] is checkMeshTool


# OCP: adding a tool needs no solver/CLI edit ----------------------------
def test_new_tool_self_registers_and_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    stub = Tool("stubTool")
    stub.build(lambda cfg: [lazy("preprocess.stubTool", lambda _c: object())])
    register_tool(stub)
    rts = resolve_tools(
        available_tools(), PreprocessConfig(tools=[{"tool": "stubTool"}])
    )
    assert [rt.name for rt in rts] == ["preprocess.stubTool"]


# cross-tool integration (retargeted from the old mesh tests) ------------
def test_pipeline_chain_threads_meshes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, block = _fake_mesh_bindings(monkeypatch)
    steps = [lazy("_foam_time", lambda _ctx: "TIME")]
    steps.extend(tool_graph_steps(detect_tools(CASE)))
    ctx = execute_initialization(steps)
    assert calls == ["block", "snappy", "check"]
    assert ctx.mesh is block


def test_scrambled_pipeline_publishes_sink_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, block = _fake_mesh_bindings(monkeypatch)
    cfg = PreprocessConfig(
        tools=[
            {"tool": "checkMesh", "depends_on": ["snappyHexMesh"]},
            {"tool": "snappyHexMesh", "depends_on": ["blockMesh"]},
            {"tool": "blockMesh"},
        ]
    )
    steps = [lazy("_foam_time", lambda _ctx: "TIME")]
    steps.extend(tool_graph_steps(resolve_tools(available_tools(), cfg)))
    ctx = execute_initialization(steps)
    assert calls == ["block", "snappy", "check"]
    assert ctx.mesh is block


def test_failing_checkmesh_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_mesh_bindings(monkeypatch)
    monkeypatch.setattr(
        check_mesh,
        "checkMesh",
        lambda *a, **k: {"passed": False, "total_errors": 2},
    )
    steps = [lazy("_foam_time", lambda _ctx: "TIME")]
    steps.extend(tool_graph_steps(detect_tools(CASE)))
    with pytest.raises(InitStepExecutionError) as ei:
        execute_initialization(steps)
    assert ei.value.step_name == "preprocess.checkMesh"


# migration grep guards (needles are string-concatenated) ----------------
def _grep(*needles: str) -> list[str]:
    roots = [SRC, TEST]
    return [
        f"{py}: {n}"
        for root in roots
        for py in root.rglob("*.py")
        for n in needles
        if n in py.read_text()
    ]


def test_no_legacy_graph_refs() -> None:
    assert (
        _grep(
            "resolve" + "_pipeline",
            "tool_init" + "_steps",
            "framework/tools/" + "pipeline",
        )
        == []
    )


def test_mesh_module_removed() -> None:
    assert not (SRC / "tools" / "mesh.py").exists()


def test_no_solverspec_tools() -> None:
    spec = (SRC / "framework" / "solver" / "spec.py").read_text()
    assert "detect_preprocess" + "_tools" not in spec
    assert "def tools" + "(" not in spec
    assert "self." + "_tools" not in spec


def test_run_preprocess_not_in_incompressiblefluid() -> None:
    inc = (
        SRC / "solver" / "incompressibleFluid" / "incompressibleFluid.py"
    ).read_text()
    assert "run" + "_preprocess" not in inc
    assert ".tools(" not in inc


def test_no_openfoam_skip_marker_refs() -> None:
    assert _grep("requires" + "_openfoam", "check_openfoam" + "_available") == []


def test_meshstats_feature_fully_removed() -> None:
    assert _grep("mesh" + "_stats", "MESH" + "_STATS") == []


def test_no_dead_preprocess_refs() -> None:
    dead = [
        "preprocess" + "_models",
        "Preprocess" + "Model",
        "preprocess_init" + "_steps",
    ]
    offenders = [
        f"{py}: {s}" for py in SRC.rglob("*.py") for s in dead if s in py.read_text()
    ]
    assert offenders == []
    assert not (
        SRC / "solver" / "incompressibleFluid" / "models" / "preprocess"
    ).exists()
