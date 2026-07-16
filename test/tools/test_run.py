# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Standalone preprocessing runner: argv parsing, detection, and DAG execution.

The argv/detect checks are pure logic (no ``importorskip`` needed); the OF-less
``run_preprocess`` check fakes ``pyf`` + the tool bindings to prove the runner builds
and executes the mesh DAG with no solver, fields, or time loop. The real OF case drives
the bindings end to end.
"""

import os
import types
from pathlib import Path

import pytest

import neofoam.tools.run as run_mod
from neofoam.tooling.casebuild import from_template
from neofoam.tools import block_mesh, check_mesh, snappy_hex_mesh
from neofoam.tools.block_mesh import BlockMeshStep
from neofoam.tools.check_mesh import CheckMeshStep
from neofoam.tools.run import _case_dir_from_argv, detect_tools, run_preprocess
from neofoam.tools.snappy_hex_mesh import SnappyHexMeshStep

CASE = Path(__file__).parents[1] / "solver" / "incompressibleFluid" / "preprocess_case"
UNLISTED = (
    Path(__file__).parents[1]
    / "solver"
    / "incompressibleFluid"
    / "preprocess_case_unlisted"
)


def test_case_dir_from_argv_reads_case_flag() -> None:
    assert _case_dir_from_argv(["solver", "-case", "/x"]) == "/x"


def test_case_dir_from_argv_defaults_when_absent() -> None:
    assert _case_dir_from_argv(["preprocess"]) == "."


def test_case_dir_from_argv_defaults_when_flag_dangling() -> None:
    assert _case_dir_from_argv(["-case"]) == "."


def test_detect_tools_resolves_listed() -> None:
    rts = detect_tools(CASE)
    assert [rt.name for rt in rts] == [
        "preprocess.blockMesh",
        "preprocess.snappyHexMesh",
        "preprocess.checkMesh",
    ]
    block, snappy, check = (rt.config for rt in rts)
    assert (
        isinstance(block, BlockMeshStep) and block.dict_file == "system/blockMeshDict"
    )
    assert isinstance(snappy, SnappyHexMeshStep) and snappy.overwrite is True
    assert isinstance(check, CheckMeshStep) and check.fail_on_error is True


def test_detect_tools_unlisted_dict_runs_nothing() -> None:
    # A real blockMeshDict on disk must NOT trigger preprocessing — only the
    # enable file's list activates a tool.
    assert (UNLISTED / "system" / "blockMeshDict").is_file()
    assert detect_tools(UNLISTED) == []


def test_detect_tools_absent_file_is_empty(tmp_path: Path) -> None:
    assert detect_tools(tmp_path) == []


def test_run_preprocess_runs_dag_without_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OF-less: fake pyf.Time + the three tool bindings; assert the DAG runs and the
    # sink mesh is published — exercises run_preprocess with no solver import.
    block = object()
    monkeypatch.setattr(
        run_mod,
        "pyf",
        types.SimpleNamespace(Time=lambda a: "TIME", argList=lambda v: v),
    )
    for mod in (block_mesh, snappy_hex_mesh):
        monkeypatch.setattr(
            mod,
            "pyf",
            types.SimpleNamespace(dictionary=types.SimpleNamespace(read=lambda f: f)),
        )
    monkeypatch.setattr(block_mesh, "generate_blockmesh", lambda *a, **k: block)
    monkeypatch.setattr(
        snappy_hex_mesh, "generate_snappy_hex_mesh", lambda *a, **k: None
    )
    monkeypatch.setattr(
        check_mesh, "checkMesh", lambda *a, **k: {"passed": True, "total_errors": 0}
    )
    ctx = run_preprocess(["preprocess", "-case", str(CASE)])
    assert ctx.mesh is block
    assert "U" not in ctx.fields


def test_run_preprocess_real(tmp_path: Path) -> None:
    case_dir = from_template(CASE).build_at(tmp_path / "case")
    cwd = Path.cwd()
    os.chdir(case_dir.path)
    try:
        ctx = run_preprocess(["preprocess", "-case", str(case_dir.path)])
        assert ctx.mesh.nCells() > 0
        assert "U" not in ctx.fields
    finally:
        os.chdir(cwd)


def test_run_preprocess_real_from_outside_case(tmp_path: Path) -> None:
    # run_preprocess resolves -case and chdirs into it, so preprocessing a case
    # from an unrelated cwd builds the mesh (the tools read system/blockMeshDict
    # relative to the working dir) and restores the original cwd afterwards.
    case_dir = from_template(CASE).build_at(tmp_path / "case")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    cwd = Path.cwd()
    os.chdir(outside)
    try:
        ctx = run_preprocess(["preprocess", "-case", str(case_dir.path)])
        assert ctx.mesh.nCells() > 0
        assert Path.cwd() == outside  # cwd restored
    finally:
        os.chdir(cwd)
