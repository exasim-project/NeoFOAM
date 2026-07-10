# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Fixtures for the cross-backend operator parity tests.

Per mesh the flow is: stage a case copy, generate the mesh, seed deterministic
T/U fields, then — for every div-scheme variant — rewrite ``system/fvSchemes``
and run one subprocess per backend/executor that evaluates all applicable
operators and dumps ``.npy`` results. Every solver/reader subprocess owns its
one ``Foam::Time`` (and, for neon, the Kokkos runtime); the fixtures only
orchestrate and load numpy arrays.

pybFoam, neon and the OpenFOAM binaries (blockMesh, snappyHexMesh) are hard
requirements — nothing here is guarded or skipped except GPU-executor runs on
hosts without a GPU device.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from operator_defs import DIV_SCHEMES, IMPLICIT_TWINS, fv_schemes_text, ops_for_scheme

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent.parent
SETUP_OPERATOR_CASE = REPO_ROOT / "test" / "setup_operator"
TILTED_CUBE_CASE = REPO_ROOT / "tutorials" / "tiltedCube"
CARTESIAN_OVERLAY = TEST_DIR / "cases" / "cartesian"
TILTED_CUBE_OVERLAY = TEST_DIR / "cases" / "tiltedCube"

RUN_OPENFOAM = TEST_DIR / "run_openfoam_ops.py"
RUN_NEON = TEST_DIR / "run_neon_ops.py"
SEED_FIELDS = TEST_DIR / "seed_fields.py"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["FOAM_SIGFPE"] = "false"
    return env


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 600.0) -> None:
    proc = subprocess.run(
        cmd, cwd=cwd, env=_env(), capture_output=True, text=True, timeout=timeout
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

# Meshes with empty front/back patches, where the z component of vector results
# is only defined up to the empty-patch treatment (see test/operators.cpp).
TWO_D_MESHES = {"cartesian_nx5", "cartesian_nx20"}


@dataclass
class MeshResults:
    """Operator results for one mesh: ``of[scheme][op]``, ``neon[executor][scheme][op]``."""

    of: dict[str, dict[str, np.ndarray]]
    neon: dict[str, dict[str, dict[str, np.ndarray]]]


def _compute_results(case: Path, executors: list[str]) -> MeshResults:
    of: dict[str, dict[str, np.ndarray]] = {}
    neon: dict[str, dict[str, dict[str, np.ndarray]]] = {ex: {} for ex in executors}
    results_root = case / "_results"
    for scheme in DIV_SCHEMES:
        (case / "system" / "fvSchemes").write_text(fv_schemes_text(scheme))
        ops = ops_for_scheme(scheme)
        # implicit operators are evaluated on the neon side only; their
        # pybFoam reference is the explicit twin's result
        of_ops = [op for op in ops if op not in IMPLICIT_TWINS]
        out_of = results_root / f"of_{scheme}"
        out_of.mkdir(parents=True)
        _run([sys.executable, str(RUN_OPENFOAM), str(case), str(out_of), *of_ops])
        of[scheme] = {op: np.load(out_of / f"{op}.npy") for op in of_ops}
        for executor in executors:
            out_nn = results_root / f"neon_{executor}_{scheme}"
            out_nn.mkdir(parents=True)
            _run(
                [sys.executable, str(RUN_NEON), str(case), str(out_nn), executor, *ops]
            )
            neon[executor][scheme] = {op: np.load(out_nn / f"{op}.npy") for op in ops}
    return MeshResults(of=of, neon=neon)


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Whether the neon GPU executor can be constructed on this host."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import neon._neon as nn; nn.initialize(['probe']); "
            "nn.GPUExecutor(); nn.finalize()",
        ],
        env=_env(),
        capture_output=True,
        timeout=120.0,
    )
    return proc.returncode == 0


@pytest.fixture(scope="session")
def mesh_results(
    tmp_path_factory: pytest.TempPathFactory, gpu_available: bool
) -> Callable[[str], MeshResults]:
    """Lazy per-mesh cache: stage + mesh + seed + run all backends once per mesh."""
    cache: dict[str, MeshResults] = {}
    executors = ["Serial"] + (["GPU"] if gpu_available else [])

    def get(mesh: str) -> MeshResults:
        if mesh not in cache:
            case = MESHES[mesh](tmp_path_factory.mktemp(mesh) / "case")
            cache[mesh] = _compute_results(case, executors)
        return cache[mesh]

    return get
