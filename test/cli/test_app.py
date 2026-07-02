# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""CLI surface for preprocessing: ``--no-preprocess`` gating + ``preprocess``."""

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.solver.incompressibleFluid.create_fields import create_init  # noqa: E402

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
    import neofoam.tools.run as run_mod
    from typer.testing import CliRunner

    from neofoam.cli.app import app

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


def test_preprocess_command_runs_pipeline_only(tmp_path: Path) -> None:
    from neofoam.tools.run import run_preprocess

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


def test_consume_dag_flags_sets_env_and_strips_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--dag-init PATH`` is removed from passthrough args + sets the env var."""
    from neofoam.cli.app import _consume_dag_flags
    from neofoam.framework.initialization.execution.executor import DUMP_INIT_DAG_ENV

    monkeypatch.delenv(DUMP_INIT_DAG_ENV, raising=False)
    rest = _consume_dag_flags(["-case", ".", "--dag-init", "d.dot", "-parallel"])
    assert rest == ["-case", ".", "-parallel"]  # solver args pass through untouched
    # relative path is resolved to absolute (solver chdirs into the case)
    assert os.environ[DUMP_INIT_DAG_ENV] == str((Path.cwd() / "d.dot").resolve())


def test_consume_dag_flags_accepts_equals_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neofoam.cli.app import _consume_dag_flags
    from neofoam.framework.initialization.execution.executor import DUMP_INIT_DAG_ENV

    monkeypatch.delenv(DUMP_INIT_DAG_ENV, raising=False)
    rest = _consume_dag_flags(["--dag-init=/tmp/x.txt", "-case", "."])
    assert rest == ["-case", "."]
    assert os.environ[DUMP_INIT_DAG_ENV] == "/tmp/x.txt"


def test_consume_dag_flags_missing_value_errors() -> None:
    import typer

    from neofoam.cli.app import _consume_dag_flags

    with pytest.raises(typer.BadParameter):
        _consume_dag_flags(["-case", ".", "--dag-init"])


def test_consume_dag_flags_operation_dag_and_dump_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--dag-operation PATH`` and ``--dag-only`` set their env vars."""
    from neofoam.cli.app import _consume_dag_flags

    monkeypatch.delenv("NEOFOAM_DUMP_OPERATION_DAG", raising=False)
    monkeypatch.delenv("NEOFOAM_DUMP_DAG_ONLY", raising=False)
    rest = _consume_dag_flags(
        ["-case", ".", "--dag-operation", "ops.dot", "--dag-only"]
    )
    assert rest == ["-case", "."]
    assert os.environ["NEOFOAM_DUMP_OPERATION_DAG"] == str(
        (Path.cwd() / "ops.dot").resolve()
    )
    assert os.environ["NEOFOAM_DUMP_DAG_ONLY"] == "1"
