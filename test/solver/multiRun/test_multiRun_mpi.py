# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun under ``mpirun``: each region on processes of its own, as on threads.

The case is ``heatConductionNeoN``'s ``coupledBlocks`` without its coupling: the
regions ``left`` (4 x 3 cells) and ``right`` (3 x 5), each an OpenFOAM case of
its own, decomposed into two along y. Both run ``foamDiffusion``
(``foam_solver.py``), explicit diffusion on OpenFOAM alone (NeoN cannot run on
part of the MPI processes yet). ``mpi_worker.py`` runs the case and prints each
region's cell centres and ``T`` per process; a run on processes must give the
run on threads, cell by cell (matched by centre), up to round-off: the
processor patches of a region exchange within the region's ``-world``.

Every case the run cannot take is refused before it starts, by rank 0, and the
run aborts: too few processes, a region not decomposed, regions on one process
next to regions on several (OpenFOAM's ``-world`` split would hang), a NeoN
region, a coupled case. An error in one region aborts every process.

Each test runs ``mpirun`` in a subprocess with a timeout, so a hang fails it.
The tests need mpi4py and an ``mpirun`` of the same MPI (the extra
``neofoam[mpi]``), and skip without them.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
import yaml

HERE = Path(__file__).parent
WORKER = HERE / "mpi_worker.py"
REGIONS = HERE.parent / "heatConductionNeoN" / "cases" / "coupledBlocks"
COUPLING = REGIONS / "system" / "regionCoupling.yaml"
RESULT_PREFIX = "#RESULT "
TIMEOUT = 180

DECOMPOSE_PAR_DICT = """FoamFile
{
    version 2.0; format ascii; class dictionary; object decomposeParDict;
}
numberOfSubdomains 2;
method simple;
coeffs { n (1 2 1); }
"""


def _mpirun() -> Optional[str]:
    """The ``mpirun`` next to the interpreter (its environment's MPI), else the one on PATH."""
    beside = Path(sys.executable).parent / "mpirun"
    return str(beside) if beside.exists() else shutil.which("mpirun")


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None or _mpirun() is None,
    reason="needs mpi4py and mpirun (the extra neofoam[mpi])",
)


def _environment() -> dict[str, str]:
    path = [str(HERE.parent.parent), os.environ.get("PYTHONPATH", "")]
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in path if p),
        "OMPI_MCA_rmaps_base_oversubscribe": "1",
        "FOAM_SIGFPE": "false",
    }


def _run(case: Path, command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=case,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        env=_environment(),
    )


def _threads(case: Path) -> subprocess.CompletedProcess:
    return _run(case, [sys.executable, str(WORKER)])


def _processes(case: Path, n: int, *args: str) -> subprocess.CompletedProcess:
    return _run(case, [str(_mpirun()), "-np", str(n), sys.executable, str(WORKER), *args])


def _results(run: subprocess.CompletedProcess) -> dict[str, dict[str, np.ndarray]]:
    """Each region's ``T`` by cell, ordered by cell centre, from the processes' results."""
    assert run.returncode == 0, f"stdout:\n{run.stdout[-6000:]}\nstderr:\n{run.stderr[-4000:]}"
    found: dict[str, list[Any]] = {}
    for line in run.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            for region, data in json.loads(line[len(RESULT_PREFIX) :])["regions"].items():
                found.setdefault(region, []).append(data)
    regions = {}
    for region, parts in found.items():
        centres = np.concatenate([np.asarray(part["centres"]) for part in parts])
        values = np.concatenate([np.asarray(part["T"]) for part in parts])
        order = np.lexsort(centres.round(9).T[::-1])
        regions[region] = {
            "centres": centres[order],
            "T": values[order],
            "index": [part["index"] for part in parts],
        }
    return regions


def _failure(run: subprocess.CompletedProcess) -> str:
    assert run.returncode != 0, f"the run should fail:\n{run.stdout[-4000:]}"
    return re.sub(r"\s+", " ", run.stdout + run.stderr)


@pytest.fixture(scope="module")
def meshed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The two uncoupled regions, meshed and decomposed into two each."""
    case = tmp_path_factory.mktemp("mpi") / "base"
    for region in ("left", "right"):
        shutil.copytree(REGIONS / region, case / region)
        temperature = case / region / "0" / "T"
        temperature.write_text(re.sub(r"\n\s*coupled\s+\w+;", "", temperature.read_text()))
        (case / region / "system" / "decomposeParDict").write_text(DECOMPOSE_PAR_DICT)
        for tool in ("blockMesh", "decomposePar"):
            subprocess.run(
                [tool, "-case", region],
                cwd=case,
                check=True,
                capture_output=True,
                timeout=TIMEOUT,
            )
    return case


def _case(meshed: Path, tmp_path: Path, regions: dict[str, Any]) -> Path:
    case = tmp_path / "case"
    shutil.copytree(meshed, case)
    (case / "system").mkdir()
    (case / "system" / "controlDict.yaml").write_text(
        yaml.safe_dump({"endTime": 5.0, "deltaT": 1.0, "regionSolvers": regions})
    )
    return case


def _foam(processes: int) -> dict[str, Any]:
    return {"solver": "foamDiffusion", "processes": processes}


@pytest.mark.parametrize("processes", [1, 2], ids=["1+1", "2+2"])
def test_mpi_regions_match_their_run_on_threads(
    meshed: Path, tmp_path: Path, processes: int
) -> None:
    case = _case(meshed, tmp_path, {"left": _foam(processes), "right": _foam(processes)})

    threads = _results(_threads(case))
    mpi = _results(_processes(case, 2 * processes))

    assert sorted(mpi) == ["left", "right"]
    for region in ("left", "right"):
        assert mpi[region]["index"] == [5] * processes
        np.testing.assert_allclose(mpi[region]["centres"], threads[region]["centres"], atol=1e-12)
        np.testing.assert_allclose(mpi[region]["T"], threads[region]["T"], rtol=1e-12)
        assert np.ptp(mpi[region]["T"]) > 1.0, f"{region}: T did not diffuse"


def test_mpi_refuses_a_run_on_the_wrong_number_of_processes(meshed: Path, tmp_path: Path) -> None:
    case = _case(meshed, tmp_path, {"left": _foam(2), "right": _foam(2)})

    message = _failure(_processes(case, 3))

    assert (
        "the case was started on 3 MPI processes, but the regions of system/controlDict.yaml "
        "run on 2 (left) + 2 (right) = 4; start it with mpirun -np 4"
    ) in message


def test_mpi_refuses_a_region_that_is_not_decomposed(meshed: Path, tmp_path: Path) -> None:
    case = _case(meshed, tmp_path, {"left": _foam(2), "right": _foam(2)})
    for n in range(2):
        shutil.rmtree(case / "right" / f"processor{n}")

    message = _failure(_processes(case, 4))

    assert "region right runs on 2 processes, but right/ lacks processor0, processor1" in message


def test_mpi_refuses_serial_regions_next_to_parallel_ones(meshed: Path, tmp_path: Path) -> None:
    case = _case(meshed, tmp_path, {"left": _foam(1), "right": _foam(2)})

    message = _failure(_processes(case, 3))

    assert "the regions ['right'] run on several MPI processes" in message
    assert "the run would hang" in message


def test_mpi_refuses_a_neon_region(meshed: Path, tmp_path: Path) -> None:
    case = _case(
        meshed,
        tmp_path,
        {"left": {"solver": "heatConductionNeoN", "processes": 2}, "right": _foam(2)},
    )

    message = _failure(_processes(case, 4))

    assert (
        "region left runs on 2 of the 4 MPI processes, but its solver heatConductionNeoN "
        "cannot run on part of them: NeoN communicates on MPI_COMM_WORLD"
    ) in message


def test_mpi_refuses_a_coupled_case(meshed: Path, tmp_path: Path) -> None:
    case = _case(meshed, tmp_path, {"left": _foam(1), "right": _foam(1)})
    shutil.copyfile(COUPLING, case / "system" / "regionCoupling.yaml")

    message = _failure(_processes(case, 2))

    assert "the MPI comm channel is not implemented yet" in message
    assert "interface (left - right)" in message


def test_mpi_aborts_every_process_when_one_region_fails(meshed: Path, tmp_path: Path) -> None:
    case = _case(meshed, tmp_path, {"left": _foam(2), "right": _foam(2)})

    message = _failure(_processes(case, 4, "fail:right"))

    assert re.search(r"region right \(rank [01] of 2\) failed", message)
    assert "RuntimeError: foamDiffusion: region right fails as asked" in message
