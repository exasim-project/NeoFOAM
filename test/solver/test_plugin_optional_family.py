# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end: a plugin's new optional model family loads and runs in every solver.

**Integration, real solvers.** ``_optional_family_plugin.py`` is the plugin shown in
``doc/reference/cli.rst``. It defines a family NeoFOAM does not know, whose one member
is active when the case has ``constant/radiationProperties`` (the checked-in overlay)
and whose operation is placed after ``momentum``, the one step all three loops share.

**Subprocess per run.** One ``Foam::Time`` per process, and the binding and the
registration are global: running them in a worker keeps both out of this process.

**Expected calls.** The operation sits in the outer corrector loop, so it runs once
per time step and outer corrector of the case.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from neofoam.tooling.casebuild import CaseDir, block_mesh, from_template, patch

from .incompressibleFluid.solved_case import built_cavity
from .incompressibleVoF.conftest import build_case, overlay

_HERE = Path(__file__).parent
_ENABLE_OVERLAY = _HERE / "cases" / "plugin_optional_family"
_PIMPLE_TEMPLATE = _HERE.parent / "setup_pimple"
_VOF_BOX = _HERE / "incompressibleVoF" / "cases" / "damBreakBox"


def _cavity(dest: Path) -> Path:
    return built_cavity(dest, end_time=0.002, delta_t=0.001).path


def _adaptive_cavity(dest: Path) -> Path:
    case = built_cavity(dest, end_time=0.002, delta_t=0.001)
    patch("system/controlDict", adjustTimeStep=True, maxDeltaT=0.001)(case)
    return case.path


def _neon_cavity(dest: Path) -> Path:
    # Without residualControl the outer loop cannot stop early: always nOuterCorrectors passes.
    pipeline = (
        from_template(_PIMPLE_TEMPLATE)
        | patch("system/controlDict", endTime=0.005)
        | patch("system/fvSolution", remove=["PIMPLE.residualControl"])
        | block_mesh()
    )
    return pipeline.build_at(dest).path


def _vof_box(dest: Path) -> Path:
    return build_case(
        dest,
        overlay(_VOF_BOX),
        patch("system/controlDict", endTime=0.005, deltaT=0.005),
        block_mesh(),
    )


@dataclass(frozen=True)
class _Scenario:
    solver: str
    build: Callable[[Path], Path]
    enabled: bool
    loaded: tuple[str, ...]
    calls: int
    binds_viscosity: bool


_SCENARIOS = {
    "fluid": _Scenario("incompressibleFluid", _cavity, True, ("radiationP1",), 2, True),
    "fluid_after_builtin": _Scenario(
        "incompressibleFluid", _adaptive_cavity, True, ("maxDeltaT", "radiationP1"), 2, True
    ),
    "fluid_not_enabled": _Scenario("incompressibleFluid", _cavity, False, (), 0, True),
    "neon": _Scenario("incompressibleFluidNeoN", _neon_cavity, True, ("radiationP1",), 3, False),
    "vof": _Scenario("incompressibleVoF", _vof_box, True, ("radiationP1",), 1, False),
}


@pytest.fixture(scope="module", params=_SCENARIOS, ids=list(_SCENARIOS))
def solved(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[_Scenario, dict[str, Any]]:
    """The scenario's case, run by the worker with the plugin family bound."""
    scenario = _SCENARIOS[request.param]
    case = scenario.build(tmp_path_factory.mktemp(request.param) / "case")
    if scenario.enabled:
        shutil.copytree(_ENABLE_OVERLAY, case, dirs_exist_ok=True)

    result = subprocess.run(
        [sys.executable, str(_HERE / "_optional_family_worker.py"), scenario.solver],
        cwd=case,
        env={**os.environ, "FOAM_SIGFPE": "false"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    return scenario, json.loads((CaseDir(case).path / "plugin_family.json").read_text())


def test_plugin_family_model_loads_in_family_order(
    solved: tuple[_Scenario, dict[str, Any]],
) -> None:
    scenario, result = solved
    assert tuple(result["loaded"]) == scenario.loaded


def test_plugin_family_operation_runs_in_the_solver_loop(
    solved: tuple[_Scenario, dict[str, Any]],
) -> None:
    scenario, result = solved
    assert result["calls"] == scenario.calls


def test_plugin_family_member_joins_the_solver_catalog(
    solved: tuple[_Scenario, dict[str, Any]],
) -> None:
    _, result = solved
    assert "radiationP1" in result["catalog"]


def test_existing_family_member_reaches_solvers_binding_that_family(
    solved: tuple[_Scenario, dict[str, Any]],
) -> None:
    scenario, result = solved
    assert ("Carreau" in result["catalog"]) == scenario.binds_viscosity
