# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The heated plate under ``mpirun``: the fluid on 1 or 2 processes, the plate on 1.

``cases/heatedPlateOnProcesses/fluid<n>`` lays the decomposition and the
processes over ``cases/heatedPlate``. OpenFOAM writes a region on one process
as a serial case, on more into ``processor<N>/``. The fluid's written wall
temperature is the probe's, as on threads (same ``ATOL_K``).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from _mpirun import mpirun, needs_mpi

from neofoam.solver.incompressibleFluid.coupled_patches import PATCH_TYPES
from neofoam.solver.regions import coupled_entries
from solver.incompressibleFluid._heated_plate import (
    PROBE_WALL_T,
    mesh_copy,
    run,
    written_interface_T,
)
from solver.incompressibleFluid.test_heated_plate import ATOL_K

pytestmark = needs_mpi

#: the fluid's processes -> its written parts
PARTS = {1: ["."], 2: ["processor0", "processor1"]}


@pytest.fixture(scope="module", params=sorted(PARTS), ids=["1+1", "2+1"])
def plate(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, list[Path]]:
    processes = request.param
    case = mesh_copy(
        tmp_path_factory.mktemp("heatedPlate") / "case",
        f"heatedPlateOnProcesses/fluid{processes}",
        tools=("blockMesh", "decomposePar"),
    )
    command = [str(mpirun()), "-np", str(processes + 1), sys.executable, "-m"]
    run([*command, "neofoam.solver.multiRun"], case)
    return case, [case / "fluid" / part for part in PARTS[processes]]


def test_heated_plate_mpi_written_parts_restart_coupled(plate: tuple[Path, list[Path]]) -> None:
    for part in plate[1]:
        written = part / "0.03" / "T"

        assert coupled_entries(written, PATCH_TYPES) == {"interface": "coupledTemperature"}, part


def test_heated_plate_mpi_written_wall_temperature_is_the_probes(
    plate: tuple[Path, list[Path]],
) -> None:
    case, parts = plate
    staged = [case / f"staged{n}" for n in range(len(parts))]
    wall_T = np.concatenate(
        [written_interface_T(case / "fluid", part, to) for part, to in zip(parts, staged)]
    )

    assert len(wall_T) == 161
    np.testing.assert_allclose(
        (wall_T.mean(), wall_T.min(), wall_T.max()),
        PROBE_WALL_T,
        rtol=0,
        atol=ATOL_K,
        err_msg=f"written wall T on {len(parts)} part(s)",
    )
