# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Fixtures for the cross-backend operator parity tests.

Tests import the two backend singletons (``pyb`` — pybFoam/OpenFOAM,
``nb`` — neon/NeoN) from ``backends`` and take the staged-case fields
(``T``, ``U``, ``Gamma``, ``phi``) as fixtures, which bind them to the
test's ``mesh`` and ``executor`` params. Every backend call returns a plain
numpy array; the asserts live in the test body.

Per mesh the case is staged once per session: the checked-in inputs
(``cases/common`` plus ``cases/<mesh>/blockMeshDict``) are copied to a temp
dir, and the pybFoam worker generates the mesh and seeds the deterministic
``T``/``U`` fields in-process at startup (``pybFoam.meshing`` — no external
OpenFOAM binaries). Each backend/executor then gets one persistent worker
subprocess on that case (one ``Foam::Time`` / Kokkos runtime per process —
see ``backends.py``); the pybFoam worker always starts first, since the neon
workers read the mesh and fields it wrote.

pybFoam and neon are hard requirements — nothing here is guarded or skipped
except GPU-executor runs on hosts without a GPU device.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import backends
import pytest
from backends import Field, Worker

TEST_DIR = Path(__file__).parent
CASES_DIR = TEST_DIR / "cases"

OF_WORKER = TEST_DIR / "of_worker.py"
NEON_WORKER = TEST_DIR / "neon_worker.py"

MESH_NAMES = ["cartesian_nx5", "cartesian_nx20", "sheared"]
EXECUTORS = ["Serial", "GPU"]


def _stage_case(mesh: str, dest: Path) -> Path:
    """Copy the checked-in case inputs; the of_worker meshes and seeds on top."""
    shutil.copytree(CASES_DIR / "common", dest)
    shutil.copytree(dest / "0.orig", dest / "0")
    shutil.copy(CASES_DIR / mesh / "blockMeshDict", dest / "system")
    return dest


class WorkerPool:
    """Stages each mesh once and keeps one worker per backend/case/executor.

    The pybFoam worker is always started (and awaited) first — it generates
    the mesh and seeds the fields that the neon workers read from disk.
    """

    def __init__(self, tmp_path_factory: pytest.TempPathFactory):
        self._tmp = tmp_path_factory
        self._cases: dict[str, Path] = {}
        self._workers: dict[tuple[str, ...], Worker] = {}

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


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Whether the neon GPU executor can be constructed on this host."""
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
    return proc.returncode == 0


def _bound_field(
    name: str, request: pytest.FixtureRequest, gpu_available: bool
) -> Field:
    """Bind a staged-case field to the test's ``mesh``/``executor`` params."""
    params = request.node.callspec.params
    if params["executor"] == "GPU" and not gpu_available:
        pytest.skip("no GPU executor available on this host")
    return Field(name, params["mesh"], params["executor"])


@pytest.fixture
def T(request: pytest.FixtureRequest, gpu_available: bool) -> Field:
    """Seeded ``0/T``: ``2 + sin(pi x) cos(pi y) + 0.3 z`` (see of_worker.py)."""
    return _bound_field("T", request, gpu_available)


@pytest.fixture
def U(request: pytest.FixtureRequest, gpu_available: bool) -> Field:
    """Seeded ``0/U`` with divergence in every direction (see of_worker.py)."""
    return _bound_field("U", request, gpu_available)


@pytest.fixture
def Gamma(request: pytest.FixtureRequest, gpu_available: bool) -> Field:
    """Uniform diffusivity 1: volume field on pybFoam, surface field on neon."""
    return _bound_field("Gamma", request, gpu_available)


@pytest.fixture
def phi(request: pytest.FixtureRequest, gpu_available: bool) -> Field:
    """Face flux derived from ``U`` (``fvc::flux(U)`` — no ``0/phi`` is staged)."""
    return _bound_field("phi", request, gpu_available)
