# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun under ``mpirun``: each region on processes of its own, as on threads.

The case is ``heatConductionNeoN``'s ``coupledBlocks`` without its coupling: the
regions ``left`` (4 x 3 cells) and ``right`` (3 x 5), each an OpenFOAM case of
its own, decomposed into as many parts as it has processes (along y; one part:
``processor0``, every region runs ``-parallel -world <region>``). They run
``foamDiffusion`` (``foam_solver.py``), explicit diffusion on OpenFOAM alone,
or ``heatConductionNeoN``, whose NeoN communicates on the region's world.
``mpi_worker.py`` runs the case and prints each region's cell centres and ``T``
per process; a run on processes must give the run on threads, cell by cell
(matched by centre), up to round-off (OpenFOAM) or the linear solver's
tolerance (NeoN): the processor patches of a region exchange within the
region's world.

Coupled, ``coupledBlocks`` runs as on threads too: its regions' roots exchange
the coupling over the MPI comm channel, each root gathering its region's part of
the interface (decomposed along y, the interface spans the region's
processes) and scattering what the processes receive. Its fields and the
interface data match the run on threads up to the linear solver's tolerance.

Every case the run cannot take is refused before it starts, by rank 0, and the
run aborts: too few processes, a region not decomposed (also one on one
process, without ``processor0``), and a NeoN region if this NeoN build
communicates on MPI_COMM_WORLD only. An error in one region aborts every
process, also of a coupled case, where the other region's root waits for it.

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
{{
    version 2.0; format ascii; class dictionary; object decomposeParDict;
}}
numberOfSubdomains {n};
method simple;
coeffs {{ n (1 {n} 1); }}
"""

#: The initial temperature of both regions, in K.
INITIAL_T = 350.0

#: The numbers of processes a region of the tests runs on, each decomposed in advance.
PARTS = (1, 2)


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


def _by_point(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``points`` and ``values`` (one row each), ordered by point."""
    order = np.lexsort(points.round(9).T[::-1])
    return points[order], values[order]


def _results(run: subprocess.CompletedProcess) -> dict[str, dict[str, Any]]:
    """Each region's ``T`` by cell, ordered by cell centre, from the processes' results.

    And per coupled side (``sides``, by label) the Array data it sends, ordered by point.
    """
    assert run.returncode == 0, f"stdout:\n{run.stdout[-6000:]}\nstderr:\n{run.stderr[-4000:]}"
    found: dict[str, list[Any]] = {}
    for line in run.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            for region, data in json.loads(line[len(RESULT_PREFIX) :])["regions"].items():
                found.setdefault(region, []).append(data)
    regions = {}
    for region, parts in found.items():
        centres, values = _by_point(
            np.concatenate([np.asarray(part["centres"]) for part in parts]),
            np.concatenate([np.asarray(part["T"]) for part in parts]),
        )
        sides: dict[str, dict[str, np.ndarray]] = {}
        for label in parts[0]["sides"]:
            points = np.concatenate(
                [np.asarray(part["sides"][label]["points"]).reshape(-1, 3) for part in parts]
            )
            sides[label] = {}
            for name in parts[0]["sides"][label]:
                if name != "points":
                    data = np.concatenate(
                        [np.asarray(part["sides"][label][name]) for part in parts]
                    )
                    sides[label]["points"], sides[label][name] = _by_point(points, data)
        regions[region] = {
            "centres": centres,
            "T": values,
            "index": [part["index"] for part in parts],
            "sides": sides,
        }
    return regions


def _failure(run: subprocess.CompletedProcess) -> str:
    assert run.returncode != 0, f"the run should fail:\n{run.stdout[-4000:]}"
    return re.sub(r"\s+", " ", run.stdout + run.stderr)


def _neon_takes_a_communicator() -> bool:
    """Whether the NeoN of the tests' interpreter runs a region on part of the processes."""
    probe = "from neofoam.solver.neon_runtime import neon_takes_a_communicator as f; print(f())"
    found = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=TIMEOUT
    )
    return found.stdout.strip().endswith("True")


@pytest.fixture(scope="module")
def meshed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The two uncoupled regions, meshed, in ``<n>/`` decomposed into n parts for each of PARTS."""
    base = tmp_path_factory.mktemp("mpi") / "base"
    for parts in PARTS:
        case = base / str(parts)
        for region in ("left", "right"):
            shutil.copytree(REGIONS / region, case / region)
            temperature = case / region / "0" / "T"
            temperature.write_text(re.sub(r"\n\s*coupled\s+\w+;", "", temperature.read_text()))
            (case / region / "system" / "decomposeParDict").write_text(
                DECOMPOSE_PAR_DICT.format(n=parts)
            )
            for tool in ("blockMesh", "decomposePar"):
                subprocess.run(
                    [tool, "-case", region],
                    cwd=case,
                    check=True,
                    capture_output=True,
                    timeout=TIMEOUT,
                )
    return base


@pytest.fixture(scope="module")
def coupled(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``coupledBlocks`` with its coupling, meshed, in ``<n>/`` decomposed into n parts (PARTS)."""
    base = tmp_path_factory.mktemp("mpi_coupled") / "base"
    for parts in PARTS:
        case = base / str(parts)
        for region in ("left", "right"):
            shutil.copytree(REGIONS / region, case / region)
            (case / region / "system" / "decomposeParDict").write_text(
                DECOMPOSE_PAR_DICT.format(n=parts)
            )
            for tool in ("blockMesh", "decomposePar"):
                subprocess.run(
                    [tool, "-case", region],
                    cwd=case,
                    check=True,
                    capture_output=True,
                    timeout=TIMEOUT,
                )
    return base


def _coupled_case(coupled: Path, tmp_path: Path, left: int, right: int) -> Path:
    """``coupledBlocks`` on ``left`` + ``right`` processes, its times and coupling as they are."""
    case = tmp_path / "case"
    for region, processes in (("left", left), ("right", right)):
        shutil.copytree(coupled / str(processes) / region, case / region)
    (case / "system").mkdir()
    times = yaml.safe_load((REGIONS / "system" / "controlDict.yaml").read_text())
    times["regionSolvers"] = {"left": _neon(left), "right": _neon(right)}
    (case / "system" / "controlDict.yaml").write_text(yaml.safe_dump(times))
    shutil.copyfile(COUPLING, case / "system" / "regionCoupling.yaml")
    return case


def _processes_of(entry: Any) -> int:
    return 1 if isinstance(entry, str) else entry.get("processes", 1)


def _case(meshed: Path, tmp_path: Path, regions: dict[str, Any]) -> Path:
    """The case of ``regions``: each region decomposed into its processes."""
    case = tmp_path / "case"
    for region, entry in regions.items():
        shutil.copytree(meshed / str(_processes_of(entry)) / region, case / region)
    (case / "system").mkdir()
    (case / "system" / "controlDict.yaml").write_text(
        yaml.safe_dump({"endTime": 5.0, "deltaT": 1.0, "regionSolvers": regions})
    )
    return case


def _foam(processes: int) -> dict[str, Any]:
    return {"solver": "foamDiffusion", "processes": processes}


def _neon(processes: int) -> dict[str, Any]:
    return {"solver": "heatConductionNeoN", "processes": processes}


def _assert_matches(
    mpi: dict[str, Any], threads: dict[str, Any], layout: dict[str, int], rtol: float
) -> None:
    assert sorted(mpi) == sorted(layout)
    for region, processes in layout.items():
        assert mpi[region]["index"] == [5] * processes
        np.testing.assert_allclose(mpi[region]["centres"], threads[region]["centres"], atol=1e-12)
        np.testing.assert_allclose(mpi[region]["T"], threads[region]["T"], rtol=rtol)
        assert np.abs(mpi[region]["T"] - INITIAL_T).max() > 1.0, f"{region}: T did not change"


@pytest.mark.parametrize(("left", "right"), [(1, 1), (1, 2), (2, 2)], ids=["1+1", "1+2", "2+2"])
def test_mpi_regions_match_their_run_on_threads(
    meshed: Path, tmp_path: Path, left: int, right: int
) -> None:
    case = _case(meshed, tmp_path, {"left": _foam(left), "right": _foam(right)})

    threads = _results(_threads(case))
    mpi = _results(_processes(case, left + right))

    _assert_matches(mpi, threads, {"left": left, "right": right}, rtol=1e-12)


@pytest.mark.parametrize(("left", "right"), [(1, 2), (2, 2)], ids=["1+2", "2+2"])
def test_mpi_neon_regions_match_their_run_on_threads(
    meshed: Path, tmp_path: Path, left: int, right: int
) -> None:
    if not _neon_takes_a_communicator():
        pytest.skip("this NeoN build communicates on MPI_COMM_WORLD only")
    case = _case(meshed, tmp_path, {"left": _neon(left), "right": _neon(right)})

    threads = _results(_threads(case))
    mpi = _results(_processes(case, left + right))

    _assert_matches(mpi, threads, {"left": left, "right": right}, rtol=1e-9)


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


def test_mpi_refuses_a_region_on_one_process_without_processor0(
    meshed: Path, tmp_path: Path
) -> None:
    case = _case(meshed, tmp_path, {"left": _foam(1), "right": _foam(2)})
    shutil.rmtree(case / "left" / "processor0")

    message = _failure(_processes(case, 3))

    assert "region left runs on 1 process, but left/ lacks processor0" in message


def test_mpi_refuses_a_neon_region_if_neon_communicates_on_the_world(
    meshed: Path, tmp_path: Path
) -> None:
    if _neon_takes_a_communicator():
        pytest.skip("this NeoN build runs on the region's communicator")
    case = _case(meshed, tmp_path, {"left": _neon(2), "right": _foam(2)})

    message = _failure(_processes(case, 4))

    assert (
        "region left runs on 2 of the 4 MPI processes, but its solver heatConductionNeoN "
        "cannot run on part of them: this NeoN build communicates on MPI_COMM_WORLD"
    ) in message


@pytest.mark.parametrize(("left", "right"), [(1, 1), (1, 2), (2, 2)], ids=["1+1", "1+2", "2+2"])
def test_mpi_coupled_regions_match_their_run_on_threads(
    coupled: Path, tmp_path: Path, left: int, right: int
) -> None:
    if not _neon_takes_a_communicator():
        pytest.skip("this NeoN build communicates on MPI_COMM_WORLD only")
    case = _coupled_case(coupled, tmp_path, left, right)

    threads = _results(_threads(case))
    mpi = _results(_processes(case, left + right))

    for region, processes in (("left", left), ("right", right)):
        assert mpi[region]["index"] == [threads[region]["index"][0]] * processes
        np.testing.assert_allclose(mpi[region]["centres"], threads[region]["centres"], atol=1e-12)
        np.testing.assert_allclose(mpi[region]["T"], threads[region]["T"], rtol=1e-9)
        assert np.abs(mpi[region]["T"] - INITIAL_T).max() > 1.0, f"{region}: T did not change"
    # the interface: left's wall temperature, right's heat flux into it, face by face
    for region, label, name in (("left", "left.interface", "T"), ("right", "right.interface", "q")):
        there, here = mpi[region]["sides"][label], threads[region]["sides"][label]
        np.testing.assert_allclose(there["points"], here["points"], atol=1e-12)
        np.testing.assert_allclose(there[name], here[name], rtol=1e-9, atol=1e-9)
    # the interface spans the processes: 3 (left) and 5 (right) faces, split along y
    assert len(mpi["left"]["sides"]["left.interface"]["T"]) == 3
    assert len(mpi["right"]["sides"]["right.interface"]["q"]) == 5


def test_mpi_aborts_every_process_when_a_coupled_region_fails(
    coupled: Path, tmp_path: Path
) -> None:
    if not _neon_takes_a_communicator():
        pytest.skip("this NeoN build communicates on MPI_COMM_WORLD only")
    case = _coupled_case(coupled, tmp_path, 2, 2)

    # rank 3: right's second process, not its root; left's root waits for right's
    message = _failure(_processes(case, 4, "fail-coupled:right:3"))

    assert "region right (rank 1 of 2) failed" in message
    assert "RuntimeError: region right fails on rank 3 as asked" in message


def test_mpi_aborts_every_process_when_one_region_fails(meshed: Path, tmp_path: Path) -> None:
    case = _case(meshed, tmp_path, {"left": _foam(2), "right": _foam(2)})

    message = _failure(_processes(case, 4, "fail:right"))

    assert re.search(r"region right \(rank [01] of 2\) failed", message)
    assert "RuntimeError: foamDiffusion: region right fails as asked" in message
