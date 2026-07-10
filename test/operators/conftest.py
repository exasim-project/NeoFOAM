# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Fixtures for the cross-backend operator parity tests.

Tests import the two backend singletons (``pyb`` — pybFoam/OpenFOAM,
``nb`` — neon/NeoN) from ``backends`` and take the staged-case fields
(``T``, ``U``, ``Gamma``, ``phi``) as fixtures, which bind them to the
test's ``mesh`` and ``executor`` params. Every backend call returns a plain
numpy array; the asserts live in the test body.

Per mesh the case is staged once per session: copy, generate the mesh
(blockMesh, plus the snappyHexMesh pipeline for tiltedCube), seed the
deterministic ``0/`` fields, write the shared ``system/fvSchemes``. Each
backend/executor then gets one persistent worker subprocess on that case
(one ``Foam::Time`` / Kokkos runtime per process — see ``backends.py``).

pybFoam, neon and the OpenFOAM binaries (blockMesh, snappyHexMesh) are hard
requirements — nothing here is guarded or skipped except GPU-executor runs
on hosts without a GPU device.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterator

import backends
import pytest
from backends import Field, Worker
from schemes import FV_SCHEMES

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent
SETUP_OPERATOR_CASE = REPO_ROOT / "test" / "setup_operator"
TILTED_CUBE_CASE = REPO_ROOT / "tutorials" / "tiltedCube"
CARTESIAN_OVERLAY = TEST_DIR / "cases" / "cartesian"
TILTED_CUBE_OVERLAY = TEST_DIR / "cases" / "tiltedCube"

OF_WORKER = TEST_DIR / "of_worker.py"
NEON_WORKER = TEST_DIR / "neon_worker.py"
SEED_FIELDS = TEST_DIR / "seed_fields.py"

MESH_NAMES = ["cartesian_nx5", "cartesian_nx20", "tiltedCube"]
EXECUTORS = ["Serial", "GPU"]

# Meshes with empty front/back patches, where the z component of vector
# results is only defined up to the empty-patch treatment (see
# test/operators.cpp).
TWO_D_MESHES = {"cartesian_nx5", "cartesian_nx20"}


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 600.0) -> None:
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


def _seed(case: Path) -> None:
    _run([sys.executable, str(SEED_FIELDS), str(case)])


def _stage_cartesian(dest: Path, nx: int) -> Path:
    """Copy test/setup_operator, set NX, overlay complete 0/ fields, mesh, seed."""
    shutil.copytree(SETUP_OPERATOR_CASE, dest)
    for name in ("T", "U", "Gamma"):
        shutil.copy(CARTESIAN_OVERLAY / "0.orig" / name, dest / "0" / name)
    params = dest / "system" / "simulationParameters"
    params.write_text(
        re.sub(r"NX\s+\d+;", f"NX              {nx};", params.read_text())
    )
    _run(["blockMesh"], cwd=dest)
    _seed(dest)
    return dest


def _stage_tilted_cube(dest: Path) -> Path:
    """Copy tutorials/tiltedCube, run its snappy pipeline, overlay fields, seed.

    The Allmesh script needs OpenFOAM's RunFunctions, so its steps are replayed
    directly: STL generation, blockMesh, surfaceFeatureExtract, snappyHexMesh.
    """
    shutil.copytree(TILTED_CUBE_CASE, dest)
    _run([sys.executable, "makeTiltedCube.py"], cwd=dest / "constant" / "triSurface")
    _run(["blockMesh"], cwd=dest)
    _run(["surfaceFeatureExtract"], cwd=dest)
    _run(["snappyHexMesh", "-overwrite"], cwd=dest, timeout=1800.0)
    zero = dest / "0"
    zero.mkdir(exist_ok=True)
    for name in ("T", "U", "Gamma"):
        shutil.copy(TILTED_CUBE_OVERLAY / "0.orig" / name, zero / name)
    _seed(dest)
    return dest


MESHES: dict[str, Callable[[Path], Path]] = {
    "cartesian_nx5": lambda dest: _stage_cartesian(dest, 5),
    "cartesian_nx20": lambda dest: _stage_cartesian(dest, 20),
    "tiltedCube": _stage_tilted_cube,
}


class WorkerPool:
    """Stages each mesh once and keeps one worker per backend/case/executor."""

    def __init__(self, tmp_path_factory: pytest.TempPathFactory):
        self._tmp = tmp_path_factory
        self._cases: dict[str, Path] = {}
        self._workers: dict[tuple[str, ...], Worker] = {}

    def _case(self, mesh: str) -> Path:
        if mesh not in self._cases:
            case = MESHES[mesh](self._tmp.mktemp(mesh) / "case")
            (case / "system" / "fvSchemes").write_text(FV_SCHEMES)
            self._cases[mesh] = case
        return self._cases[mesh]

    def of_worker(self, mesh: str) -> Worker:
        key = ("of", mesh)
        if key not in self._workers:
            case = self._case(mesh)
            self._workers[key] = Worker(OF_WORKER, case, "of", argv=[])
        return self._workers[key]

    def neon_worker(self, mesh: str, executor: str) -> Worker:
        key = ("neon", mesh, executor)
        if key not in self._workers:
            case = self._case(mesh)
            self._workers[key] = Worker(
                NEON_WORKER, case, f"neon_{executor}", argv=[executor]
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
    """Seeded ``0/T``: ``2 + sin(pi x) cos(pi y) + 0.3 z`` (see seed_fields.py)."""
    return _bound_field("T", request, gpu_available)


@pytest.fixture
def U(request: pytest.FixtureRequest, gpu_available: bool) -> Field:
    """Seeded ``0/U`` with in-plane divergence (see seed_fields.py)."""
    return _bound_field("U", request, gpu_available)


@pytest.fixture
def Gamma(request: pytest.FixtureRequest, gpu_available: bool) -> Field:
    """Uniform diffusivity 1: volume field on pybFoam, surface field on neon."""
    return _bound_field("Gamma", request, gpu_available)


@pytest.fixture
def phi(request: pytest.FixtureRequest, gpu_available: bool) -> Field:
    """Face flux derived from ``U`` (``fvc::flux(U)`` — no ``0/phi`` is staged)."""
    return _bound_field("phi", request, gpu_available)
