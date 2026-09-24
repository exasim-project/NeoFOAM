# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""CLI surface for preprocessing: ``--no-preprocess`` gating + ``preprocess``."""

import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

import neofoam.tools.run as run_mod
from neofoam.cli.app import app
from neofoam.solver.incompressibleFluid.create_fields import create_init
from neofoam.tools.run import run_preprocess

CASE = Path(__file__).parents[1] / "solver" / "incompressibleFluid" / "preprocess_case"


def test_pipeline_detected_without_flag() -> None:
    cwd = Path.cwd()
    os.chdir(CASE)
    try:
        runner = create_init(CASE)
        runner.argv = ["incompressibleFluid"]
        runner.run_load()
    finally:
        os.chdir(cwd)
    assert [rt.name for rt in runner.preprocess_tools] == [
        "preprocess.blockMesh",
        "preprocess.snappyHexMesh",
        "preprocess.checkMesh",
    ]


def test_no_preprocess_flag_skips_detection() -> None:
    cwd = Path.cwd()
    os.chdir(CASE)
    try:
        runner = create_init(CASE)
        runner.argv = ["incompressibleFluid", "--no-preprocess"]
        runner.run_load()
    finally:
        os.chdir(cwd)
    assert runner.preprocess_tools == []


def test_preprocess_command_wires_case_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run_preprocess(argv: "list[str] | None" = None) -> object:
        captured["argv"] = list(argv or [])
        return object()

    # The CLI does ``from neofoam.tools.run import run_preprocess`` at call time,
    # so patching the module attribute takes effect.
    monkeypatch.setattr(run_mod, "run_preprocess", fake_run_preprocess)
    result = CliRunner().invoke(app, ["preprocess", "/some/case"])
    assert result.exit_code == 0
    assert captured["argv"][1:] == ["-case", "/some/case"]


def test_solver_command_rejects_postprocess_with_a_clear_error() -> None:
    """No neofoam solver implements OpenFOAM's ``-postProcess`` mode (running
    the case's registered function objects without solving) — pybFoam exposes
    no functionObject-execution binding to build it on. The CLI must fail fast
    with a one-line message instead of falling through to pybFoam's
    ``argList``, which prints a confusing raw usage dump for an unknown flag.
    """
    result = CliRunner().invoke(app, ["solver", "incompressiblevof", "-postProcess", "-time", "0"])

    assert result.exit_code == 1
    assert "solver -postProcess mode not implemented" in result.output


def test_preprocess_command_runs_pipeline_only(tmp_path: Path) -> None:
    case = tmp_path / "case"
    shutil.copytree(CASE, case)
    cwd = Path.cwd()
    os.chdir(case)
    try:
        ctx = run_preprocess(["preprocess", "-case", str(case)])
        assert ctx.mesh.nCells() > 0
        # No solver operations ran: no fields were built onto the Context.
        assert "U" not in ctx.fields
    finally:
        os.chdir(cwd)
