# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""CLI surface for preprocessing: ``--no-preprocess`` gating + ``preprocess``."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.solver.incompressibleFluid.create_fields import create_init  # noqa: E402


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
    import neofoam.solver.incompressibleFluid as inc
    from typer.testing import CliRunner

    from neofoam.cli.app import app

    captured: dict[str, list[str]] = {}

    def fake_run_preprocess(argv: "list[str] | None" = None) -> object:
        captured["argv"] = list(argv or [])
        return object()

    monkeypatch.setattr(inc, "run_preprocess", fake_run_preprocess)
    result = CliRunner().invoke(app, ["preprocess", "/some/case"])
    assert result.exit_code == 0
    assert captured["argv"][1:] == ["-case", "/some/case"]


@requires_openfoam
def test_preprocess_command_runs_pipeline_only(tmp_path: Path) -> None:
    from neofoam.solver.incompressibleFluid import run_preprocess

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
