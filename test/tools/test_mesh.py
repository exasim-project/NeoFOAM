# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared mesh tools: checkMesh raise/pass (unit) + in-process pipeline (OF).

The pure-Python unit checks monkeypatch ``checkMesh`` so no mesh is built. The
``@requires_openfoam`` cases drive the real bindings against the
``preprocess_case`` fixture (no ``constant/polyMesh`` on disk).
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pybFoam")

from neofoam.framework.initialization import InitStepExecutionError, lazy  # noqa: E402
from neofoam.framework.initialization.execution import (  # noqa: E402
    execute_initialization,
)
from neofoam.framework.tools import (  # noqa: E402
    PreprocessConfig,
    ToolRuntime,
    resolve_pipeline,
    tool_init_steps,
)
from neofoam.solver.incompressibleFluid import incompressibleFluid  # noqa: E402
from neofoam.tools import mesh  # noqa: E402
from neofoam.tools.mesh import (  # noqa: E402
    BlockMeshStep,
    CheckMeshStep,
    SnappyHexMeshStep,
    blockMeshTool,
    checkMeshTool,
    snappyHexMeshTool,
)


def _openfoam_available() -> bool:
    try:
        return (
            subprocess.run(
                ["blockMesh", "-help"], capture_output=True, timeout=5
            ).returncode
            == 0
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


requires_openfoam = pytest.mark.skipif(
    not _openfoam_available(), reason="OpenFOAM not available"
)

CASE = Path(__file__).parents[1] / "solver" / "incompressibleFluid" / "preprocess_case"
UNLISTED = (
    Path(__file__).parents[1]
    / "solver"
    / "incompressibleFluid"
    / "preprocess_case_unlisted"
)


def _staged_case(tmp_path: Path) -> Path:
    """Copy the (polyMesh-free) fixture into ``tmp_path`` so generated mesh
    files land outside the committed tree."""
    assert not (CASE / "constant" / "polyMesh").exists()
    dest = tmp_path / "case"
    shutil.copytree(CASE, dest)
    return dest


def _check_step(**cfg_kwargs: Any) -> Any:
    cfg = CheckMeshStep(tool="checkMesh", **cfg_kwargs)
    runtime = ToolRuntime(spec=checkMeshTool, name="preprocess.checkMesh", config=cfg)
    return runtime.run_build()[0]


def test_tools_import_from_shared_package() -> None:
    # The capability now lives in the shared package, not under a solver.
    assert blockMeshTool.name == "blockMesh"
    assert snappyHexMeshTool.name == "snappyHexMesh"
    assert checkMeshTool.name == "checkMesh"
    assert BlockMeshStep is not None
    assert SnappyHexMeshStep is not None
    assert CheckMeshStep is not None


def test_incompressiblefluid_exposes_three_tools() -> None:
    assert blockMeshTool in incompressibleFluid._tools
    assert snappyHexMeshTool in incompressibleFluid._tools
    assert checkMeshTool in incompressibleFluid._tools


def test_checkmesh_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mesh,
        "checkMesh",
        lambda *a, **k: {"passed": False, "total_errors": 3},
    )
    step = _check_step(fail_on_error=True)
    with pytest.raises(InitStepExecutionError) as excinfo:
        step.initializer({"_prev_mesh": object()})
    assert excinfo.value.step_name == "preprocess.checkMesh"


def test_checkmesh_passes_returns_prior_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    stats = {"passed": True, "total_errors": 0}
    monkeypatch.setattr(mesh, "checkMesh", lambda *a, **k: stats)
    step = _check_step(fail_on_error=True)
    prior = object()
    # checkMesh only validates; it passes the prior mesh straight through.
    assert step.initializer({"_prev_mesh": prior}) is prior


def test_checkmesh_does_not_raise_when_fail_on_error_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats = {"passed": False, "total_errors": 5}
    monkeypatch.setattr(mesh, "checkMesh", lambda *a, **k: stats)
    step = _check_step(fail_on_error=False)
    prior = object()
    assert step.initializer({"_prev_mesh": prior}) is prior


def test_checkmesh_wraps_binding_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = RuntimeError("binding boom")

    def raise_runtime(*a: Any, **k: Any) -> Any:
        raise boom

    monkeypatch.setattr(mesh, "checkMesh", raise_runtime)
    step = _check_step(fail_on_error=True)
    with pytest.raises(InitStepExecutionError) as excinfo:
        step.initializer({"_prev_mesh": object()})
    assert excinfo.value.step_name == "preprocess.checkMesh"
    assert excinfo.value.__cause__ is boom


@pytest.mark.parametrize("exc_type", [ValueError, TypeError])
def test_checkmesh_reraises_value_or_type_error(
    monkeypatch: pytest.MonkeyPatch, exc_type: type[Exception]
) -> None:
    boom = exc_type("bad checkMesh argument")

    def raise_it(*a: Any, **k: Any) -> Any:
        raise boom

    monkeypatch.setattr(mesh, "checkMesh", raise_it)
    step = _check_step(fail_on_error=True)
    with pytest.raises(exc_type) as excinfo:
        step.initializer({"_prev_mesh": object()})
    assert excinfo.value is boom


def test_pipeline_chain_threads_meshes_without_openfoam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    calls: list[tuple[str, Any]] = []
    block_mesh = object()

    # blockMesh/snappy read a dict via ``pyf.dictionary.read``; swap the whole
    # ``pyf`` reference (the pybind11 ``dictionary`` type rejects attr patching).
    monkeypatch.setattr(
        mesh,
        "pyf",
        types.SimpleNamespace(dictionary=types.SimpleNamespace(read=lambda f: f)),
    )

    def fake_block(time: Any, d: Any, verbose: bool = False) -> Any:
        calls.append(("block", time))
        return block_mesh

    def fake_snappy(
        m: Any, d: Any, overwrite: bool = True, verbose: bool = True
    ) -> None:
        calls.append(("snappy", m))  # mutates in place (returns None)

    def fake_check(m: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(("check", m))
        return {"passed": True, "total_errors": 0}

    monkeypatch.setattr(mesh, "generate_blockmesh", fake_block)
    monkeypatch.setattr(mesh, "generate_snappy_hex_mesh", fake_snappy)
    monkeypatch.setattr(mesh, "checkMesh", fake_check)

    runtimes = incompressibleFluid.detect_preprocess_tools(CASE)
    steps = [lazy("_foam_time", lambda _ctx: "TIME")]
    steps.extend(tool_init_steps(runtimes))
    ctx = execute_initialization(steps)

    assert [c[0] for c in calls] == ["block", "snappy", "check"]
    assert calls[0][1] == "TIME"  # blockMesh consumed _foam_time
    assert calls[1][1] is block_mesh  # snappy refined the blockMesh result
    assert calls[2][1] is block_mesh  # checkMesh validated the threaded mesh
    assert ctx.mesh is block_mesh  # terminal alias publishes one mesh resource


def test_scrambled_pipeline_publishes_sink_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    calls: list[str] = []
    block_mesh = object()

    monkeypatch.setattr(
        mesh,
        "pyf",
        types.SimpleNamespace(dictionary=types.SimpleNamespace(read=lambda f: f)),
    )

    def fake_block(*a: Any, **k: Any) -> Any:
        calls.append("block")
        return block_mesh

    def fake_snappy(*a: Any, **k: Any) -> None:
        calls.append("snappy")

    def fake_check(*a: Any, **k: Any) -> dict[str, Any]:
        calls.append("check")
        return {"passed": True, "total_errors": 0}

    monkeypatch.setattr(mesh, "generate_blockmesh", fake_block)
    monkeypatch.setattr(mesh, "generate_snappy_hex_mesh", fake_snappy)
    monkeypatch.setattr(mesh, "checkMesh", fake_check)

    # entries deliberately scrambled — order must come from depends_on
    cfg = PreprocessConfig(
        tools=[
            {"tool": "checkMesh", "depends_on": ["snappyHexMesh"]},
            {"tool": "snappyHexMesh", "depends_on": ["blockMesh"]},
            {"tool": "blockMesh"},
        ]
    )
    runtimes = resolve_pipeline(incompressibleFluid._tools, cfg)
    steps = [lazy("_foam_time", lambda _ctx: "TIME")]
    steps.extend(tool_init_steps(runtimes))
    ctx = execute_initialization(steps)

    assert calls == ["block", "snappy", "check"]  # DAG order from scrambled file
    assert ctx.mesh is block_mesh  # sink (checkMesh) passes the block mesh through


def test_meshstats_feature_fully_removed() -> None:
    roots = [
        Path(__file__).parents[2] / "src" / "neofoam",
        Path(__file__).parents[2] / "test",
    ]
    needle_a = "mesh" + "_stats"
    needle_b = "MESH" + "_STATS"
    offenders = [
        str(py)
        for root in roots
        for py in root.rglob("*.py")
        if needle_a in py.read_text() or needle_b in py.read_text()
    ]
    assert offenders == []


def test_pipeline_propagates_configured_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    seen: dict[str, Any] = {}
    block_mesh = object()

    monkeypatch.setattr(
        mesh,
        "pyf",
        types.SimpleNamespace(dictionary=types.SimpleNamespace(read=lambda f: f)),
    )

    def fake_block(time: Any, d: Any, verbose: bool = False) -> Any:
        seen["block"] = (d, verbose)
        return block_mesh

    def fake_snappy(
        m: Any, d: Any, overwrite: bool = True, verbose: bool = True
    ) -> None:
        seen["snappy"] = (d, overwrite, verbose)

    def fake_check(m: Any, **kwargs: Any) -> dict[str, Any]:
        seen["check"] = kwargs
        return {"passed": True, "total_errors": 0}

    monkeypatch.setattr(mesh, "generate_blockmesh", fake_block)
    monkeypatch.setattr(mesh, "generate_snappy_hex_mesh", fake_snappy)
    monkeypatch.setattr(mesh, "checkMesh", fake_check)

    runtimes = [
        ToolRuntime(
            spec=blockMeshTool,
            name="preprocess.blockMesh",
            config=BlockMeshStep(tool="blockMesh", verbose=True),
        ),
        ToolRuntime(
            spec=snappyHexMeshTool,
            name="preprocess.snappyHexMesh",
            config=SnappyHexMeshStep(
                tool="snappyHexMesh", overwrite=False, verbose=True
            ),
            depends_on=["blockMesh"],
        ),
        ToolRuntime(
            spec=checkMeshTool,
            name="preprocess.checkMesh",
            config=CheckMeshStep(
                tool="checkMesh",
                all_topology=True,
                all_geometry=True,
                check_quality=True,
            ),
            depends_on=["snappyHexMesh"],
        ),
    ]
    steps = [lazy("_foam_time", lambda _ctx: "TIME")]
    steps.extend(tool_init_steps(runtimes))
    execute_initialization(steps)

    assert seen["block"] == ("system/blockMeshDict", True)
    assert seen["snappy"][0] == "system/snappyHexMeshDict"
    assert seen["snappy"][1] is False  # overwrite propagated
    assert seen["snappy"][2] is True  # verbose propagated
    assert seen["check"] == {
        "all_topology": True,
        "all_geometry": True,
        "check_quality": True,
    }


def test_failing_checkmesh_aborts_before_mesh_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    block_mesh = object()

    monkeypatch.setattr(
        mesh,
        "pyf",
        types.SimpleNamespace(dictionary=types.SimpleNamespace(read=lambda f: f)),
    )
    monkeypatch.setattr(mesh, "generate_blockmesh", lambda *a, **k: block_mesh)
    monkeypatch.setattr(mesh, "generate_snappy_hex_mesh", lambda *a, **k: None)
    monkeypatch.setattr(
        mesh, "checkMesh", lambda *a, **k: {"passed": False, "total_errors": 2}
    )

    runtimes = incompressibleFluid.detect_preprocess_tools(CASE)
    steps = [lazy("_foam_time", lambda _ctx: "TIME")]
    steps.extend(tool_init_steps(runtimes))

    with pytest.raises(InitStepExecutionError) as excinfo:
        execute_initialization(steps)
    assert excinfo.value.step_name == "preprocess.checkMesh"


def test_blockmeshdict_present_but_unlisted_runs_nothing() -> None:
    # A real blockMeshDict on disk must NOT trigger preprocessing — only the
    # enable file's pipeline drives activation.
    assert (UNLISTED / "system" / "blockMeshDict").is_file()
    rts = resolve_pipeline(
        incompressibleFluid._tools, PreprocessConfig.load(case_dir=UNLISTED)
    )
    assert rts == []


def test_detect_models_applies_typed_step_defaults() -> None:
    # Each raw pipeline entry is resolved to the tool's typed step config with
    # defaults applied — the open schema's per-entry validation.
    runtimes = resolve_pipeline(
        incompressibleFluid._tools, PreprocessConfig.load(case_dir=CASE)
    )
    block, snappy, check = (rt.config for rt in runtimes)
    assert isinstance(block, BlockMeshStep)
    assert block.dict_file == "system/blockMeshDict"
    assert block.verbose is False
    assert isinstance(snappy, SnappyHexMeshStep)
    assert snappy.overwrite is True
    assert isinstance(check, CheckMeshStep)
    assert check.fail_on_error is True
    assert check.all_topology is False


def test_no_dead_preprocess_refs() -> None:
    src = Path(__file__).parents[2] / "src" / "neofoam"
    removed = [
        "preprocess_models",
        "detect_preprocess_models",
        "_preprocess_model_specs",
        "PreprocessModel",
        "preprocess_init_steps",
    ]
    offenders: list[str] = []
    for py in src.rglob("*.py"):
        text = py.read_text()
        for symbol in removed:
            if symbol in text:
                offenders.append(f"{py}: {symbol}")
    assert offenders == []
    # the solver-local package is gone
    assert not (
        src / "solver" / "incompressibleFluid" / "models" / "preprocess"
    ).exists()


@requires_openfoam
def test_blockmesh_builds_in_process(tmp_path: Path) -> None:
    from neofoam.solver.incompressibleFluid import run_preprocess

    case = _staged_case(tmp_path)
    cwd = Path.cwd()
    os.chdir(case)
    try:
        ctx = run_preprocess(["preprocess"])
        assert ctx.mesh.nCells() > 0
    finally:
        os.chdir(cwd)


@requires_openfoam
def test_snappy_refines_prior_mesh(tmp_path: Path) -> None:
    import pybFoam as pyf
    from pybFoam.meshing import generate_blockmesh, generate_snappy_hex_mesh

    case = _staged_case(tmp_path)
    cwd = Path.cwd()
    os.chdir(case)
    try:
        time = pyf.Time(pyf.argList(["preprocess"]))
        block_mesh = generate_blockmesh(
            time, pyf.dictionary.read("system/blockMeshDict")
        )
        block_cells = block_mesh.nCells()

        generate_snappy_hex_mesh(
            block_mesh,
            pyf.dictionary.read("system/snappyHexMeshDict"),
            verbose=False,
        )
        assert block_mesh.nCells() != block_cells
    finally:
        os.chdir(cwd)


@requires_openfoam
def test_checkmesh_passes_on_valid_mesh(tmp_path: Path) -> None:
    import pybFoam as pyf
    from pybFoam.meshing import checkMesh as real_check
    from pybFoam.meshing import generate_blockmesh

    case = _staged_case(tmp_path)
    cwd = Path.cwd()
    os.chdir(case)
    try:
        time = pyf.Time(pyf.argList(["preprocess"]))
        m = generate_blockmesh(time, pyf.dictionary.read("system/blockMeshDict"))
        stats = real_check(m)
        assert stats["passed"] is True
    finally:
        os.chdir(cwd)


@requires_openfoam
def test_e2e_full_pipeline_runs_to_completion(tmp_path: Path) -> None:
    from neofoam.solver.incompressibleFluid import run

    case = _staged_case(tmp_path)
    cwd = Path.cwd()
    os.chdir(case)
    try:
        ctx = run(["preprocess"])
        assert ctx.mesh.nCells() > 0
    finally:
        os.chdir(cwd)
