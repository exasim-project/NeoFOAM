# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver registry lookups, with the installed entry points replaced by an on-disk test module.

Real installed distributions are not used: the result would depend on what else is
installed in the environment.
"""

from importlib import metadata
from pathlib import Path

import pytest

from neofoam.core.entry_points import REGION_SOLVER_GROUP
from neofoam.framework.solver.registry import list_solver_names, resolve_solver

PLUGIN_MODULES = Path(__file__).parent / "cases" / "plugin_modules"
PLUGIN_SPEC = "region_solver_test_plugin:pluginRegionSolver"


def _install(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make one region solver called ``name`` the only installed entry point."""
    installed = metadata.EntryPoint(name=name, value=PLUGIN_SPEC, group=REGION_SOLVER_GROUP)
    monkeypatch.syspath_prepend(str(PLUGIN_MODULES))
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda group: [installed] if group == REGION_SOLVER_GROUP else [],
    )


def test_list_solver_names_includes_a_plugin_region_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, "pluginRegionSolver")

    names = list_solver_names()

    assert "pluginRegionSolver" in names


def test_resolve_solver_loads_a_plugin_region_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, "pluginRegionSolver")

    spec = resolve_solver("pluginRegionSolver")

    assert spec.name == "pluginRegionSolver"


def test_resolve_solver_prefers_the_built_in_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, "incompressibleFluid")

    spec = resolve_solver("incompressibleFluid")

    assert spec.name == "incompressibleFluid"


def _install_conflicting_targets(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Install two region solvers called ``name`` with different targets."""
    installed = [
        metadata.EntryPoint(name=name, value=PLUGIN_SPEC, group=REGION_SOLVER_GROUP),
        metadata.EntryPoint(name=name, value="other_plugin:spec", group=REGION_SOLVER_GROUP),
    ]
    monkeypatch.syspath_prepend(str(PLUGIN_MODULES))
    monkeypatch.setattr(
        metadata, "entry_points", lambda group: installed if group == REGION_SOLVER_GROUP else []
    )


def test_resolve_solver_names_both_conflicting_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_conflicting_targets(monkeypatch, "pluginRegionSolver")

    with pytest.raises(ValueError, match=f"'{PLUGIN_SPEC}' and 'other_plugin:spec'"):
        resolve_solver("pluginRegionSolver")


def test_built_in_solver_wins_over_conflicting_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_conflicting_targets(monkeypatch, "incompressibleFluid")

    spec = resolve_solver("incompressibleFluid")

    assert spec.name == "incompressibleFluid"


def test_list_solver_names_lists_a_conflicting_built_in_solver_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_conflicting_targets(monkeypatch, "incompressibleFluid")

    names = list_solver_names()

    assert names.count("incompressibleFluid") == 1


def test_resolve_unknown_solver_lists_plugin_region_solvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, "pluginRegionSolver")

    with pytest.raises(ValueError, match="pluginRegionSolver"):
        resolve_solver("nope")
