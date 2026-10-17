# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``neofoam.solvers`` entry points become ``neofoam solver <name>`` commands.

The entry points are replaced by an on-disk test solver that records its argv, so
no case runs; a fresh Typer group keeps the real ``solver`` group untouched.
"""

import sys
from importlib import metadata
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from neofoam.cli.app import add_plugin_solvers
from neofoam.core.entry_points import SOLVER_GROUP

PLUGIN_MODULES = Path(__file__).parents[1] / "core" / "cases" / "plugin_modules"


def _install_solver(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make the test solver the only ``neofoam.solvers`` entry point, under ``name``."""
    monkeypatch.syspath_prepend(str(PLUGIN_MODULES))
    solver = metadata.EntryPoint(name=name, value="entry_point_test_solver:run", group=SOLVER_GROUP)
    monkeypatch.setattr(
        metadata, "entry_points", lambda group: [solver] if group == SOLVER_GROUP else []
    )


def _solver_group(*builtin: str) -> typer.Typer:
    """A ``solver`` group holding commands named ``builtin``."""
    solvers = typer.Typer()
    for name in builtin:
        solvers.command(name=name)(lambda: None)
    return solvers


def test_plugin_solver_runs_with_the_passed_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_solver(monkeypatch, "addonesolver")
    solvers = _solver_group("icofoam")
    add_plugin_solvers(solvers)

    result = CliRunner().invoke(solvers, ["addonesolver", "-case", "my_case"])

    import entry_point_test_solver  # noqa: PLC0415  (on sys.path only inside the test)

    assert result.exit_code == 0, result.output
    assert entry_point_test_solver.calls[-1] == [sys.argv[0], "-case", "my_case"]


def test_plugin_solver_may_not_shadow_a_builtin_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_solver(monkeypatch, "icofoam")

    with pytest.raises(ValueError, match="'icofoam'"):
        add_plugin_solvers(_solver_group("icofoam"))
