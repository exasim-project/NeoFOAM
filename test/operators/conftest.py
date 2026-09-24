# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Session plumbing for the cross-backend operator parity tests.

Tests build their own setup in the test body: ``simulation(mesh, executor)``
(from ``case_setup``) returns the session-cached case handle, fields are
loaded by name and seeded with numpy, and the backend singletons ``pyb`` /
``nb`` (from ``backends``) evaluate the operators. This module only provides
the parametrization constants and the session ``WorkerPool``.

Per mesh the case is staged once per session: the checked-in inputs
(``cases/common`` plus ``cases/<mesh>/blockMeshDict``) are copied to a temp
dir, and the pybFoam worker generates the mesh in-process at startup
(``pybFoam.meshing`` — no external OpenFOAM binaries). Each backend/executor
then gets one persistent worker subprocess on that case (one ``Foam::Time``
/ Kokkos runtime per process — see ``backends.py``); the pybFoam worker
always starts first, since the neon workers read the mesh and the
test-seeded fields it writes.

pybFoam and neon are hard requirements — nothing here is guarded or skipped
except GPU-executor runs on hosts without a GPU device.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import backends
import pytest
from backends import Worker

if TYPE_CHECKING:
    import numpy as np

TEST_DIR = Path(__file__).parent
CASES_DIR = TEST_DIR / "cases"

OF_WORKER = TEST_DIR / "of_worker.py"
NEON_WORKER = TEST_DIR / "neon_worker.py"

MESH_NAMES = ["cartesian_nx5", "cartesian_nx20", "sheared"]
EXECUTORS = ["Serial", "GPU"]


def _stage_case(mesh: str, dest: Path) -> Path:
    """Copy the checked-in case inputs; the of_worker meshes on top."""
    shutil.copytree(CASES_DIR / "common", dest)
    shutil.copytree(dest / "0.orig", dest / "0")
    shutil.copy(CASES_DIR / mesh / "blockMeshDict", dest / "system")
    return dest


class WorkerPool:
    """Stages each mesh once and keeps one worker per backend/case/executor.

    The pybFoam worker is always started (and awaited) first — it generates
    the mesh and writes the fields that the neon workers read from disk.
    """

    def __init__(self, tmp_path_factory: pytest.TempPathFactory):
        self._tmp = tmp_path_factory
        self._cases: dict[str, Path] = {}
        self._workers: dict[tuple[str, ...], Worker] = {}
        self._cell_centres: dict[str, "np.ndarray"] = {}
        self._gpu_available: bool | None = None

    def case_dir(self, mesh: str) -> Path:
        self.of_worker(mesh)  # ensures the case is staged
        return self._cases[mesh]

    def of_worker(self, mesh: str) -> Worker:
        key = ("of", mesh)
        if key not in self._workers:
            case = _stage_case(mesh, self._tmp.mktemp(mesh) / "case")
            worker = Worker(OF_WORKER, case, "of", argv=[])
            worker.wait_ready()
            self._cases[mesh] = case
            self._workers[key] = worker
        return self._workers[key]

    def neon_worker(self, mesh: str, executor: str) -> Worker:
        key = ("neon", mesh, executor)
        if key not in self._workers:
            self.of_worker(mesh)  # ensures the case is staged on disk
            self._workers[key] = Worker(
                NEON_WORKER, self._cases[mesh], f"neon_{executor}", argv=[executor]
            )
        return self._workers[key]

    def cell_centres(self, mesh: str) -> "np.ndarray":
        if mesh not in self._cell_centres:
            self._cell_centres[mesh] = self.of_worker(mesh).evaluate("mesh.C", ())
        return self._cell_centres[mesh]

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

    def close(self) -> None:
        for worker in self._workers.values():
            worker.close()


@pytest.fixture(scope="session", autouse=True)
def _worker_pool(tmp_path_factory: pytest.TempPathFactory) -> "Iterator[None]":
    """Install the session's WorkerPool behind the ``pyb``/``nb`` singletons."""
    backends.POOL = WorkerPool(tmp_path_factory)
    yield
    backends.POOL.close()
    backends.POOL = None
