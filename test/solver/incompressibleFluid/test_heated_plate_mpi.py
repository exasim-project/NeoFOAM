# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The heated plate of ``test_heated_plate.py`` under ``mpirun``: a region on processes of its own.

The fluid (an OpenFOAM region; NeoN is loaded for the plate) on 1 or 2
processes, the plate on 1. The fluid's wall temperature at 0.03 s, as each
process writes its part of ``T``, is the probe's, as on threads, and each
written part keeps the ``coupled`` entry of the interface.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import yaml
from foamlib import FoamFile

from neofoam.solver.incompressibleFluid.coupled_patches import coupled_entries
from solver.incompressibleFluid.test_heated_plate import CASE, PROBE_WALL_T

TIMEOUT = 600

DECOMPOSE_PAR_DICT = """FoamFile
{{
    version 2.0; format ascii; class dictionary; object decomposeParDict;
}}
numberOfSubdomains {n};
method simple;
coeffs {{ n ({n} 1 1); }}
"""


def _mpirun() -> Optional[str]:
    """The ``mpirun`` next to the interpreter (its environment's MPI), else the one on PATH."""
    beside = Path(sys.executable).parent / "mpirun"
    return str(beside) if beside.exists() else shutil.which("mpirun")


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None or _mpirun() is None,
    reason="needs mpi4py and mpirun (the extra neofoam[mpi])",
)


def _case(tmp_path: Path, fluid: int) -> Path:
    """The heated plate, meshed, the fluid decomposed into ``fluid`` parts, the plate into 1."""
    case = tmp_path / "case"
    shutil.copytree(CASE, case)
    for region, parts in (("fluid", fluid), ("solid", 1)):
        (case / region / "system" / "decomposeParDict").write_text(
            DECOMPOSE_PAR_DICT.format(n=parts)
        )
        for tool in ("blockMesh", "decomposePar"):
            subprocess.run(
                [tool, "-case", region], cwd=case, check=True, capture_output=True, timeout=TIMEOUT
            )
    times = yaml.safe_load((case / "system" / "controlDict.yaml").read_text())
    times["regionSolvers"]["fluid"] = {"solver": "incompressibleFluid", "processes": fluid}
    (case / "system" / "controlDict.yaml").write_text(yaml.safe_dump(times))
    return case


@pytest.mark.parametrize("fluid", [1, 2], ids=["1+1", "2+1"])
def test_mpi_heated_plate_is_the_probes(tmp_path: Path, fluid: int) -> None:
    case = _case(tmp_path, fluid)
    done = subprocess.run(
        [str(_mpirun()), "-np", str(fluid + 1), sys.executable, "-m", "neofoam.solver.multiRun"],
        cwd=case,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        env={**os.environ, "OMPI_MCA_rmaps_base_oversubscribe": "1", "FOAM_SIGFPE": "false"},
    )
    assert done.returncode == 0, f"stdout:\n{done.stdout[-4000:]}\nstderr:\n{done.stderr[-4000:]}"
    # OpenFOAM writes a region on one process as a serial case, on more in processor<N>/
    parts = (
        [case / "fluid" / "0.03" / "T"]
        if fluid == 1
        else [case / "fluid" / f"processor{n}" / "0.03" / "T" for n in range(fluid)]
    )
    for part in parts:
        assert coupled_entries("T", part) == {"interface": "coupledTemperature"}
    values = [FoamFile(part)["boundaryField"]["interface"]["value"] for part in parts]
    wall_T = np.concatenate([np.atleast_1d(np.asarray(v, dtype=float)) for v in values])
    assert len(wall_T) == 161
    np.testing.assert_allclose(
        (wall_T.mean(), wall_T.min(), wall_T.max()), PROBE_WALL_T, rtol=0, atol=2e-6
    )
