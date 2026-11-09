# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Flow over a heated plate, its first three time steps: the fluid coupled to the plate.

``cases/heatedPlate``: the fluid (incompressibleFluid, Boussinesq with gravity,
laminar, 12013 cells) over the plate (heatConductionNeoN, 100 x 25 cells, 310 K
at its bottom), coupled Dirichlet-Neumann on the fluid's ``interface``
(``coupledTemperature``: receives the wall temperature, sends the heat flux into
the fluid) and the plate's ``top`` (``coupledHeatFlux``), as their ``0/T``
declare; mapped nearest-neighbor (161 graded faces against 100); serial, the
fluid first, Aitken; the node wraps the fluid's PIMPLE loop (``inner_loop``) and
the plate's ``solve_T``. deltaT 0.01 s to 0.03 s.

The reference is the monkeypatched probe the coupled patch was built from
(``build/heated_plate/probe4_coupled`` of the coupling-heat worktree, same case
to 1 s): its passes per time step and its fluid wall temperature at 0.03 s.
The case is the tutorial ``tutorials/flowOverHeatedPlate`` but for its end time
and write interval (the tutorial runs to 1 s).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from foamlib import FoamFile

from neofoam.solver.incompressibleFluid.coupled_patches import coupled_entries

HERE = Path(__file__).parent
CASE = HERE / "cases" / "heatedPlate"
WORKER = HERE / "cht_worker.py"
RESULT_PREFIX = "#RESULT "
TUTORIAL = HERE.parents[2] / "tutorials" / "flowOverHeatedPlate"
#: the files of the case whose times differ from the tutorial's
CONTROL_DICTS = ("system/controlDict.yaml", "fluid/system/controlDict", "solid/system/controlDict")

#: the probe at 0.03 s: passes per time step, the fluid's wall temperature (mean, min, max)
PROBE_PASSES = [15, 16, 15]
PROBE_WALL_T = (301.315643, 300.962216, 301.347602)


@pytest.fixture(scope="module")
def plate(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    case_dir = tmp_path_factory.mktemp("heatedPlate") / "case"
    shutil.copytree(CASE, case_dir)
    for region in ("fluid", "solid"):
        subprocess.run(
            ["blockMesh", "-case", str(case_dir / region)],
            check=True,
            capture_output=True,
            timeout=120,
        )
    done = subprocess.run(
        [sys.executable, str(WORKER)],
        cwd=case_dir,
        env={**os.environ, "FOAM_SIGFPE": "false"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, (
        f"cht_worker failed (rc={done.returncode}):\n{done.stdout[-2000:]}\n{done.stderr[-3000:]}"
    )
    answers = [line for line in done.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    assert answers, f"cht_worker printed no result:\n{done.stdout[-2000:]}"
    computed = dict(json.loads(answers[-1][len(RESULT_PREFIX) :]))
    computed["case_dir"] = case_dir
    return computed


def test_both_regions_reach_the_end_time(plate: dict[str, Any]) -> None:
    assert plate["time"] == pytest.approx({"fluid": 0.03, "solid": 0.03})


def test_the_fluid_side_is_the_dirichlet_side(plate: dict[str, Any]) -> None:
    assert list(plate["fluid"]) == ["interface"]
    interface = plate["fluid"]["interface"]
    assert (interface["type"], interface["condition"]) == (
        "CoupledTemperature",
        "coupledTemperature",
    )
    points = np.asarray(interface["points"])
    assert points.shape == (161, 3)
    np.testing.assert_allclose(points[:, 1], 0.0)  # the face centres on the plate's top


def test_the_passes_and_the_wall_temperature_are_the_probes(plate: dict[str, Any]) -> None:
    assert plate["passes"] == {"solid": PROBE_PASSES}  # the leader: the last region
    wall_T = np.asarray(plate["fluid"]["interface"]["wall_T"])
    np.testing.assert_allclose(
        (wall_T.mean(), wall_T.min(), wall_T.max()), PROBE_WALL_T, rtol=0, atol=2e-6
    )


def test_the_heat_through_the_interface_is_the_same_on_both_sides(plate: dict[str, Any]) -> None:
    fluid = plate["fluid"]["interface"]
    into_fluid = float(np.dot(fluid["q"], fluid["areas"]))
    out_of_plate = float(np.dot(plate["solid"]["q_out"], plate["solid"]["areas"]))
    assert into_fluid > 0
    # nearest-neighbor between non-matching faces conserves only to about 0.1 %
    assert into_fluid == pytest.approx(out_of_plate, rel=2e-3)


def test_the_written_temperature_restarts_coupled(plate: dict[str, Any]) -> None:
    # OpenFOAM's fixedValue drops the coupled entry; the writer adds it back
    written = plate["case_dir"] / "fluid" / "0.03" / "T"
    assert coupled_entries("T", written) == {"interface": "coupledTemperature"}
    np.testing.assert_allclose(
        _patch_values(written, "interface"), plate["fluid"]["interface"]["wall_T"], rtol=1e-9
    )


def _patch_values(path: Path, patch: str) -> Any:
    return np.asarray(FoamFile(path)["boundaryField"][patch]["value"], dtype=float)


def test_the_tutorial_is_this_case_run_to_1_s() -> None:
    files = sorted(path.relative_to(CASE) for path in CASE.rglob("*") if path.is_file())
    assert files and all(
        (TUTORIAL / name).read_bytes() == (CASE / name).read_bytes()
        for name in files
        if str(name) not in CONTROL_DICTS
    )
    for name in CONTROL_DICTS:
        tutorial, case = (_times(root / name) for root in (TUTORIAL, CASE))
        assert (tutorial.pop("endTime"), tutorial.pop("writeInterval")) == (1, 20)
        assert (case.pop("endTime"), case.pop("writeInterval")) == (0.03, 3)
        assert tutorial == case


def _times(path: Path) -> dict[str, Any]:
    if path.suffix == ".yaml":
        return dict(yaml.safe_load(path.read_text()))
    return {key: value for key, value in FoamFile(path).as_dict().items() if key != "FoamFile"}
