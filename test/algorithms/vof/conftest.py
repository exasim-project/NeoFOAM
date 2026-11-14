# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Session plumbing of the VoF parity tests: parametrization and the worker pool.

Per mesh the ``sharpSphere`` case is staged once per session into a temp dir,
with that mesh's ``blockMeshDict`` and ``sphereCentre``. The pybFoam worker
generates the mesh in process and runs setFields and setExprFields on it, so it
always starts first; the neon workers read what it wrote. Each backend/executor
gets one persistent worker (one ``Foam::Time`` / Kokkos runtime per process).

The pool is a plain fixture, not autouse: other tests in this directory start
no workers and never see it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterator

import pytest

from ._parity_backends import _Mesh, _Simulation, _Worker

_TEST_DIR = Path(__file__).parent
_CASE = _TEST_DIR / "cases" / "sharpSphere"
_OF_WORKER = _TEST_DIR / "_of_parity_worker.py"
_NEON_WORKER = _TEST_DIR / "_neon_parity_worker.py"
_TWO_PHASE_CASE = _TEST_DIR / "cases" / "twoPhase"

MESH_NAMES = ["cartesian_nx5", "cartesian_nx20", "sheared"]
EXECUTORS = ["Serial", "CPU", "GPU"]


def _stage_case(mesh: str, dest: Path) -> Path:
    """Copy the case inputs and this mesh's dictionaries; the pybFoam worker meshes on top."""
    shutil.copytree(_CASE, dest, ignore=shutil.ignore_patterns("meshes"))
    shutil.copytree(dest / "0.orig", dest / "0")
    for name in ("blockMeshDict", "sphereCentre"):
        shutil.copy(_CASE / "meshes" / mesh / name, dest / "system")
    return dest


def _stage_two_phase_case(mesh: str, dest: Path) -> Path:
    """Copy the two-phase inputs and the mesh, its own or the matching ``sharpSphere`` one."""
    shutil.copytree(_TWO_PHASE_CASE, dest, ignore=shutil.ignore_patterns("meshes"))
    shutil.copytree(dest / "0.orig", dest / "0")
    own_mesh = _TWO_PHASE_CASE / "meshes" / mesh
    mesh_dir = own_mesh if own_mesh.is_dir() else _CASE / "meshes" / mesh
    shutil.copy(mesh_dir / "blockMeshDict", dest / "system")
    return dest


class _WorkerPool:
    """Stages each mesh once and keeps one worker per backend/case/executor."""

    _stage = staticmethod(_stage_case)
    _of_script = _OF_WORKER
    _neon_script = _NEON_WORKER

    def __init__(self, tmp_path_factory: pytest.TempPathFactory):
        self._tmp = tmp_path_factory
        self._cases: dict[str, Path] = {}
        self._workers: dict[tuple[str, ...], _Worker] = {}
        self._gpu_available: bool | None = None

    def case_dir(self, mesh: str) -> Path:
        self.of_worker(mesh)  # ensures the case is staged
        return self._cases[mesh]

    def of_worker(self, mesh: str) -> _Worker:
        key = ("of", mesh)
        if key not in self._workers:
            case = self._stage(mesh, self._tmp.mktemp(mesh) / "case")
            worker = _Worker(self._of_script, case, "of", argv=[])
            worker.wait_ready()
            self._cases[mesh] = case
            self._workers[key] = worker
        return self._workers[key]

    def neon_worker(self, mesh: str, executor: str) -> _Worker:
        key = ("neon", mesh, executor)
        if key not in self._workers:
            self.of_worker(mesh)  # ensures the case is staged on disk
            self._workers[key] = _Worker(
                self._neon_script, self._cases[mesh], f"neon_{executor}", argv=[executor]
            )
        return self._workers[key]

    def gpu_available(self) -> bool:
        """Whether the neon GPU executor can be constructed on this host."""
        if self._gpu_available is None:
            env = {**os.environ, "FOAM_SIGFPE": "false"}
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import neon._neon as nn; nn.initialize(['probe']); "
                    "nn.GPUExecutor(); nn.finalize()",
                ],
                env=env,
                capture_output=True,
                timeout=120.0,
            )
            self._gpu_available = proc.returncode == 0
        return self._gpu_available

    def simulation(self, mesh: str, executor: str) -> _Simulation:
        """The session-cached case for ``mesh`` x ``executor``; skips an unavailable GPU."""
        if executor == "GPU" and not self.gpu_available():
            pytest.skip("no GPU executor available on this host")
        worker = self.of_worker(mesh)
        geometry = _Mesh(worker.evaluate("mesh.C", ()), worker.evaluate("mesh.V", ()))
        return _Simulation(self, mesh, executor, geometry)

    def close(self) -> None:
        for worker in self._workers.values():
            worker.close()


class _TwoPhaseWorkerPool(_WorkerPool):
    """The worker pool of the ``twoPhase`` case."""

    _stage = staticmethod(_stage_two_phase_case)
    _of_script = _TEST_DIR / "_of_two_phase_worker.py"
    _neon_script = _TEST_DIR / "_neon_two_phase_worker.py"


@pytest.fixture(scope="session")
def parity_simulation(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Callable[[str, str], _Simulation]]:
    """Factory of ``(mesh, executor)`` case handles on the session's worker pool."""
    pool = _WorkerPool(tmp_path_factory)
    yield pool.simulation
    pool.close()


@pytest.fixture(scope="module")
def two_phase_simulation(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Callable[[str, str], _Simulation]]:
    """Factory of two-phase case handles; module scope frees its workers after the tests."""
    pool = _TwoPhaseWorkerPool(tmp_path_factory)
    yield pool.simulation
    pool.close()
