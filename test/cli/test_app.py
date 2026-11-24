# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""CLI surface: ``--no-preprocess`` gating, ``preprocess`` and plugin loading.

Plugin entry points are replaced by an on-disk test solver that records its argv, so
no case runs; a fresh Typer group keeps the real ``solver`` group untouched.
"""

import os
import shutil
import sys
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import neofoam.tools.run as run_mod
from neofoam.cli.app import _add_plugin_solvers, app
from neofoam.core.entry_points import PLUGIN_GROUP, SOLVER_GROUP
from neofoam.solver.incompressibleFluid.create_fields import create_init
from neofoam.tools.run import run_preprocess

CASE = Path(__file__).parents[1] / "solver" / "incompressibleFluid" / "preprocess_case"
PLUGIN_MODULES = Path(__file__).parent / "cases" / "plugin_modules"


@pytest.fixture
def plugin_modules(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Put the test solver module on the path and forget it afterwards."""
    monkeypatch.syspath_prepend(str(PLUGIN_MODULES))
    yield
    sys.modules.pop("entry_point_test_solver", None)


def _install_solvers(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make the test solver the only ``neofoam.solvers`` entry point, under each of ``names``."""
    installed = [
        metadata.EntryPoint(name=name, value="entry_point_test_solver:run", group=SOLVER_GROUP)
        for name in names
    ]
    monkeypatch.setattr(
        metadata, "entry_points", lambda group: installed if group == SOLVER_GROUP else []
    )


def _solver_group(*builtin: str) -> typer.Typer:
    """A ``solver`` group holding commands named ``builtin``."""
    solvers = typer.Typer()
    for name in builtin:
        solvers.command(name=name)(lambda: None)
    return solvers


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


@pytest.mark.usefixtures("plugin_modules")
def test_cli_callback_loads_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = metadata.EntryPoint(name="test", value="entry_point_test_solver", group=PLUGIN_GROUP)
    monkeypatch.setattr(
        metadata, "entry_points", lambda group: [plugin] if group == PLUGIN_GROUP else []
    )

    result = CliRunner().invoke(app, ["solver", "--help"])

    assert result.exit_code == 0, result.output
    assert "entry_point_test_solver" in sys.modules


@pytest.mark.usefixtures("plugin_modules")
def test_plugin_solver_runs_with_the_passed_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_solvers(monkeypatch, "addonesolver")
    solvers = _solver_group("icofoam")
    _add_plugin_solvers(solvers)

    result = CliRunner().invoke(solvers, ["addonesolver", "-case", "my_case"])

    import entry_point_test_solver  # noqa: PLC0415  (on sys.path only inside the test)

    assert result.exit_code == 0, result.output
    assert entry_point_test_solver.calls[-1] == [sys.argv[0], "-case", "my_case"]


@pytest.mark.usefixtures("plugin_modules")
def test_plugin_solver_is_registered_under_its_lower_cased_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_solvers(monkeypatch, "addOneSolver")
    solvers = _solver_group("icofoam")
    _add_plugin_solvers(solvers)

    result = CliRunner().invoke(solvers, ["addonesolver"])

    assert result.exit_code == 0, result.output


def test_plugin_solver_that_fails_to_import_names_its_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = metadata.EntryPoint(name="broken", value="no_such_module:run", group=SOLVER_GROUP)
    monkeypatch.setattr(
        metadata, "entry_points", lambda group: [broken] if group == SOLVER_GROUP else []
    )
    solvers = _solver_group("icofoam")
    _add_plugin_solvers(solvers)

    result = CliRunner().invoke(solvers, ["broken"])

    assert isinstance(result.exception, RuntimeError)
    assert "'broken' (no_such_module:run) failed to load" in str(result.exception)


@pytest.mark.parametrize("name", ["icofoam", "icoFoam"])
def test_plugin_solver_may_not_shadow_a_builtin_solver(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    _install_solvers(monkeypatch, name)

    with pytest.raises(ValueError, match="'icofoam'"):
        _add_plugin_solvers(_solver_group("icofoam"))


def test_plugin_solvers_may_not_differ_only_by_case(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_solvers(monkeypatch, "addOneSolver", "addonesolver")

    with pytest.raises(ValueError, match="'addonesolver'"):
        _add_plugin_solvers(_solver_group())
