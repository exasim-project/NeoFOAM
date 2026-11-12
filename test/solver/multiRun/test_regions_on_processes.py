# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun under ``mpirun``: each region on processes of its own gives its run on threads.

The regions are ``heatConductionNeoN``'s ``coupledBlocks`` (4 x 3 and 3 x 5 cells),
decomposed along y so the interface spans every process of a region; ``uncoupled``
drops the coupling. Cells are matched by centre. Tolerances: OpenFOAM's explicit
diffusion differs by round-off only (rtol 1e-12); NeoN and the coupled interface by
the PCG tolerance 1e-12 of ``T`` amplified over the coupling iterations (rtol 1e-9).
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from _mpirun import mpirun, needs_mpi

HERE = Path(__file__).parent
WORKER = HERE / "mpi_worker.py"
BLOCKS = HERE.parent / "heatConductionNeoN" / "cases" / "coupledBlocks"
CASES = HERE / "cases" / "regions_on_processes"
RESULT_PREFIX = "#RESULT "
TIMEOUT = 180
# K: the initial T of both regions
INITIAL_T = 350.0
# the time steps of uncoupled (endTime 5, deltaT 1) and coupledBlocks (2e6, 1e6)
STEPS = {"uncoupled": 5, "coupled": 2}
PARTS = (1, 2)

pytestmark = needs_mpi


def _run(case: Path, command: list[str]) -> subprocess.CompletedProcess:
    path = [str(HERE.parent.parent), os.environ.get("PYTHONPATH", "")]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in path if p),
        "OMPI_MCA_rmaps_base_oversubscribe": "1",
        "FOAM_SIGFPE": "false",
    }
    return subprocess.run(
        command, cwd=case, capture_output=True, text=True, timeout=TIMEOUT, env=env
    )


def _threads(case: Path) -> subprocess.CompletedProcess:
    return _run(case, [sys.executable, str(WORKER)])


def _processes(case: Path, n: int, *args: str) -> subprocess.CompletedProcess:
    return _run(case, [str(mpirun()), "-np", str(n), sys.executable, str(WORKER), *args])


def _by_point(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort(points.round(9).T[::-1])
    return points[order], values[order]


def _port(parts: list[dict[str, Any]], label: str) -> dict[str, np.ndarray]:
    """A coupled port's data from every process, ordered by point."""
    points = np.concatenate([np.reshape(part[label]["points"], (-1, 3)) for part in parts])
    port = {}
    for name in parts[0][label]:
        if name != "points":
            data = np.concatenate([np.asarray(part[label][name]) for part in parts])
            port["points"], port[name] = _by_point(points, data)
    return port


def _results(run: subprocess.CompletedProcess) -> dict[str, dict[str, Any]]:
    """Each region's ``T`` by cell centre, time index per process and coupled ports."""
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
        ports = [part["ports"] for part in parts]
        regions[region] = {
            "centres": centres,
            "T": values,
            "index": [part["index"] for part in parts],
            "ports": {label: _port(ports, label) for label in ports[0]},
        }
    return regions


def _failure(run: subprocess.CompletedProcess) -> str:
    assert run.returncode != 0, f"the run should fail:\n{run.stdout[-4000:]}"
    return re.sub(r"\s+", " ", run.stdout + run.stderr)


def _meshed(base: Path, *overlays: str) -> Path:
    """coupledBlocks' regions with ``overlays``, meshed and decomposed in ``<n>/`` (PARTS)."""
    for parts in PARTS:
        case = base / str(parts)
        for region in ("left", "right"):
            shutil.copytree(BLOCKS / region, case / region)
            for overlay in (*overlays, f"parts{parts}"):
                if (CASES / overlay / region).is_dir():
                    shutil.copytree(CASES / overlay / region, case / region, dirs_exist_ok=True)
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
def uncoupled(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _meshed(tmp_path_factory.mktemp("uncoupled"), "uncoupled")


@pytest.fixture(scope="module")
def coupled(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _meshed(tmp_path_factory.mktemp("coupled"))


def _case(meshed: Path, tmp_path: Path, control: Path, regions: dict[str, Any]) -> Path:
    """The case of ``regions`` (``{solver, processes}`` each) with the times of ``control``."""
    case = tmp_path / "case"
    for region, entry in regions.items():
        shutil.copytree(meshed / str(entry["processes"]) / region, case / region)
    shutil.copytree(control.parent, case / "system")
    times = yaml.safe_load(control.read_text())
    times["participants"] = regions
    (case / "system" / "controlDict.yaml").write_text(yaml.safe_dump(times))
    return case


def _uncoupled_case(meshed: Path, tmp_path: Path, regions: dict[str, Any]) -> Path:
    return _case(meshed, tmp_path, CASES / "uncoupled" / "system" / "controlDict.yaml", regions)


def _coupled_case(meshed: Path, tmp_path: Path, left: int, right: int) -> Path:
    regions = {"left": _neon(left), "right": _neon(right)}
    return _case(meshed, tmp_path, BLOCKS / "system" / "controlDict.yaml", regions)


def _foam(processes: int) -> dict[str, Any]:
    return {"solver": "foamDiffusion", "processes": processes}


def _neon(processes: int) -> dict[str, Any]:
    return {"solver": "heatConductionNeoN", "processes": processes}


def _assert_matches(
    mpi: dict[str, Any], threads: dict[str, Any], layout: dict[str, int], rtol: float, steps: int
) -> None:
    assert sorted(mpi) == sorted(layout)
    for region, processes in layout.items():
        case = f"{region} on {processes} processes"
        assert threads[region]["index"] == [steps], case
        assert mpi[region]["index"] == [steps] * processes, case
        np.testing.assert_allclose(
            mpi[region]["centres"], threads[region]["centres"], rtol=0, atol=1e-12, err_msg=case
        )
        np.testing.assert_allclose(
            mpi[region]["T"], threads[region]["T"], rtol=rtol, atol=0, err_msg=case
        )
        assert np.abs(mpi[region]["T"] - INITIAL_T).max() > 1.0, f"{case}: T did not change"


@pytest.mark.parametrize(("left", "right"), [(1, 1), (1, 2), (2, 2)], ids=["1+1", "1+2", "2+2"])
def test_regions_on_processes_match_their_run_on_threads(
    uncoupled: Path, tmp_path: Path, left: int, right: int
) -> None:
    case = _uncoupled_case(uncoupled, tmp_path, {"left": _foam(left), "right": _foam(right)})

    threads = _results(_threads(case))
    mpi = _results(_processes(case, left + right))

    _assert_matches(
        mpi, threads, {"left": left, "right": right}, rtol=1e-12, steps=STEPS["uncoupled"]
    )


@pytest.mark.parametrize(("left", "right"), [(1, 2), (2, 2)], ids=["1+2", "2+2"])
def test_neon_regions_on_processes_match_their_run_on_threads(
    uncoupled: Path, tmp_path: Path, left: int, right: int
) -> None:
    case = _uncoupled_case(uncoupled, tmp_path, {"left": _neon(left), "right": _neon(right)})

    threads = _results(_threads(case))
    mpi = _results(_processes(case, left + right))

    _assert_matches(
        mpi, threads, {"left": left, "right": right}, rtol=1e-9, steps=STEPS["uncoupled"]
    )


@pytest.mark.parametrize(("left", "right"), [(1, 1), (1, 2), (2, 2)], ids=["1+1", "1+2", "2+2"])
def test_coupled_regions_on_processes_match_their_run_on_threads(
    coupled: Path, tmp_path: Path, left: int, right: int
) -> None:
    case = _coupled_case(coupled, tmp_path, left, right)

    threads = _results(_threads(case))
    mpi = _results(_processes(case, left + right))

    _assert_matches(mpi, threads, {"left": left, "right": right}, rtol=1e-9, steps=STEPS["coupled"])
    # left's wall temperature and right's heat flux into it: 3 and 5 faces split along y
    for region, label, name, faces in (
        ("left", "left.interface", "T", 3),
        ("right", "right.interface", "q", 5),
    ):
        there, here = mpi[region]["ports"][label], threads[region]["ports"][label]
        case_name = f"{label} {name}, {left}+{right} processes"
        assert len(there[name]) == faces, case_name
        np.testing.assert_allclose(there["points"], here["points"], atol=1e-12, err_msg=case_name)
        np.testing.assert_allclose(there[name], here[name], rtol=1e-9, atol=1e-9, err_msg=case_name)


def test_run_refuses_the_wrong_number_of_processes(uncoupled: Path, tmp_path: Path) -> None:
    case = _uncoupled_case(uncoupled, tmp_path, {"left": _foam(2), "right": _foam(2)})

    message = _failure(_processes(case, 3))

    assert (
        "the case was started on 3 MPI processes, but the regions of system/controlDict.yaml "
        "run on 2 (left) + 2 (right) = 4; start it with mpirun -np 4"
    ) in message


def test_run_refuses_a_region_that_is_not_decomposed(uncoupled: Path, tmp_path: Path) -> None:
    case = _uncoupled_case(uncoupled, tmp_path, {"left": _foam(2), "right": _foam(2)})
    for n in range(2):
        shutil.rmtree(case / "right" / f"processor{n}")

    message = _failure(_processes(case, 4))

    assert "region right runs on 2 processes, but right/ lacks processor0, processor1" in message


def test_run_refuses_a_region_on_one_process_without_processor0(
    uncoupled: Path, tmp_path: Path
) -> None:
    case = _uncoupled_case(uncoupled, tmp_path, {"left": _foam(1), "right": _foam(2)})
    shutil.rmtree(case / "left" / "processor0")

    message = _failure(_processes(case, 3))

    assert "region left runs on 1 process, but left/ lacks processor0" in message


def test_run_aborts_every_process_when_a_coupled_follower_fails(
    coupled: Path, tmp_path: Path
) -> None:
    case = _coupled_case(coupled, tmp_path, 2, 2)

    # rank 3 is right's second process; left's root waits for right's
    message = _failure(_processes(case, 4, "fail-coupled:right:3"))

    assert "region right (rank 1 of 2) failed" in message
    assert "RuntimeError: region right fails on rank 3 as asked" in message


def test_run_aborts_every_process_when_one_region_fails(uncoupled: Path, tmp_path: Path) -> None:
    case = _uncoupled_case(uncoupled, tmp_path, {"left": _foam(2), "right": _foam(2)})

    message = _failure(_processes(case, 4, "fail:right"))

    assert re.search(r"region right \(rank [01] of 2\) failed", message)
    assert "RuntimeError: foamDiffusion: region right fails as asked" in message
