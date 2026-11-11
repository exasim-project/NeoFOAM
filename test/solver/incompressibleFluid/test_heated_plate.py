# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Flow over a heated plate, its first three time steps: the fluid coupled to the plate.

``cases/heatedPlate`` is ``tutorials/flowOverHeatedPlate`` to 0.03 s: the fluid
(incompressibleFluid, Boussinesq) over the plate (heatConductionNeoN),
Dirichlet-Neumann on the fluid's ``interface`` and the plate's ``top``, 161
against 100 faces mapped nearest-neighbor, serial with Aitken. The reference
is the probe the coupled patch was built from (same case, same arithmetic):
its passes and wall temperatures, printed to 1e-6 K, so ``ATOL_K`` covers that
rounding. Nearest-neighbor between non-matching faces does not conserve the
heat: 0.07 % at 0.03 s, ``HEAT_RTOL`` bounds it.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybFoam as pyf
import pytest
import yaml

from neofoam.solver.incompressibleFluid.coupled_patches import PATCH_TYPES
from neofoam.solver.regions import coupled_entries
from solver.incompressibleFluid._heated_plate import (
    CASE,
    HERE,
    PROBE_PASSES,
    PROBE_WALL_T,
    mesh_copy,
    result,
    written_interface_T,
)

TUTORIAL = HERE.parents[2] / "tutorials" / "flowOverHeatedPlate"
CONTROL_DICTS = ("system/controlDict.yaml", "fluid/system/controlDict", "solid/system/controlDict")
#: endTime, writeInterval
TUTORIAL_TIMES, CASE_TIMES = (1, 20), (0.03, 3)
WORD_ENTRIES = (
    "application",
    "startFrom",
    "stopAt",
    "writeControl",
    "writeFormat",
    "writeCompression",
    "timeFormat",
    "runTimeModifiable",
    "adjustTimeStep",
)
ATOL_K = 2e-6
HEAT_RTOL = 2e-3


@pytest.fixture(scope="module")
def plate(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    case_dir = mesh_copy(tmp_path_factory.mktemp("heatedPlate") / "case", tools=("blockMesh",))
    computed = result([sys.executable, str(HERE / "cht_worker.py")], case_dir)
    fluid = case_dir / "fluid"
    computed["written_T"] = written_interface_T(fluid, fluid, case_dir.parent / "staged")
    computed["case_dir"] = case_dir
    return computed


def _mean_min_max(values: Any) -> tuple[float, float, float]:
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def test_heated_plate_regions_reach_the_end_time(plate: dict[str, Any]) -> None:
    np.testing.assert_allclose(
        [plate["time"]["fluid"], plate["time"]["solid"]], 0.03, rtol=1e-12, err_msg="end times"
    )


def test_heated_plate_fluid_is_the_dirichlet_side(plate: dict[str, Any]) -> None:
    assert list(plate["fluid"]) == ["interface"]
    interface = plate["fluid"]["interface"]
    points = np.asarray(interface["points"])

    assert interface["type"] == "CoupledTemperature"
    assert points.shape == (161, 3)
    # the face centres on the plate's top, y = 0 exactly in blockMesh's vertices
    np.testing.assert_allclose(points[:, 1], 0.0, rtol=0, atol=1e-12, err_msg="interface y")


def test_heated_plate_passes_are_the_probes(plate: dict[str, Any]) -> None:
    # the leader: the last region
    assert plate["passes"] == {"solid": PROBE_PASSES}


def test_heated_plate_wall_temperature_is_the_probes(plate: dict[str, Any]) -> None:
    wall_T = plate["fluid"]["interface"]["wall_T"]

    np.testing.assert_allclose(
        _mean_min_max(wall_T), PROBE_WALL_T, rtol=0, atol=ATOL_K, err_msg="fluid wall T"
    )


def test_heated_plate_heat_through_the_interface_is_nearly_the_same_on_both_sides(
    plate: dict[str, Any],
) -> None:
    fluid = plate["fluid"]["interface"]
    into_fluid = float(np.dot(fluid["q"], fluid["areas"]))
    out_of_plate = float(np.dot(plate["solid"]["q_out"], plate["solid"]["areas"]))

    assert into_fluid > 0
    np.testing.assert_allclose(into_fluid, out_of_plate, rtol=HEAT_RTOL, err_msg="interface heat")


def test_heated_plate_written_temperature_is_the_probes(plate: dict[str, Any]) -> None:
    # writePrecision 10: 1e-7 K at 300 K
    np.testing.assert_allclose(
        _mean_min_max(plate["written_T"]), PROBE_WALL_T, rtol=0, atol=ATOL_K, err_msg="written T"
    )


def test_heated_plate_written_temperature_restarts_coupled(plate: dict[str, Any]) -> None:
    written = plate["case_dir"] / "fluid" / "0.03" / "T"

    assert plate["kept_entries"] == {"T": {"interface": "coupledTemperature"}}
    assert coupled_entries(written, PATCH_TYPES) == {"interface": "coupledTemperature"}


def test_heated_plate_case_files_are_the_tutorials() -> None:
    files = sorted(path.relative_to(CASE) for path in CASE.rglob("*") if path.is_file())
    differ = [
        str(name)
        for name in files
        if str(name) not in CONTROL_DICTS
        and (TUTORIAL / name).read_bytes() != (CASE / name).read_bytes()
    ]

    assert files and differ == []


def _entries(path: Path) -> dict[str, Any]:
    control = pyf.dictionary.read(str(path))
    return {
        name: control.get_word(name) if name in WORD_ENTRIES else control.get_scalar(name)
        for name in map(str, control.toc())
        if name != "FoamFile"
    }


@pytest.mark.parametrize("name", CONTROL_DICTS)
def test_heated_plate_is_the_tutorial_to_0_03_s(name: str) -> None:
    read = (
        (lambda path: dict(yaml.safe_load(path.read_text())))
        if name.endswith(".yaml")
        else _entries
    )
    tutorial, case = read(TUTORIAL / name), read(CASE / name)

    assert (tutorial.pop("endTime"), tutorial.pop("writeInterval")) == TUTORIAL_TIMES, name
    assert (case.pop("endTime"), case.pop("writeInterval")) == CASE_TIMES, name
    assert tutorial == case, name
