# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Mesh a copy of a heatConductionNeoN case and run it through ``heat_worker.py``."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from foamlib import FoamFile

HERE = Path(__file__).parent
CASES = HERE / "cases"
WORKER = HERE / "heat_worker.py"
RESULT_PREFIX = "#RESULT "


def mesh_copy(case: str, dest: Path) -> Path:
    """Copy ``cases/<case>`` to ``dest`` and blockMesh it (each region of a multi-region case)."""
    shutil.copytree(CASES / case, dest)
    for blockmesh_dict in sorted(dest.glob("**/system/blockMeshDict")):
        region = blockmesh_dict.parent.parent
        subprocess.run(
            ["blockMesh", "-case", str(region)], check=True, capture_output=True, timeout=120
        )
    return dest


def run_worker(case_dir: Path, mode: str) -> subprocess.CompletedProcess[str]:
    """Run the worker on ``case_dir``; FOAM_SIGFPE off, as for the other NeoN solver tests."""
    return subprocess.run(
        [sys.executable, str(WORKER), mode],
        cwd=case_dir,
        env={**os.environ, "FOAM_SIGFPE": "false"},
        capture_output=True,
        text=True,
        timeout=600,
    )


def result(case_dir: Path, mode: str) -> dict[str, Any]:
    """What the worker computed on ``case_dir``; it must succeed."""
    done = run_worker(case_dir, mode)
    assert done.returncode == 0, (
        f"heat_worker {mode} failed (rc={done.returncode}):\n"
        f"{done.stdout[-2000:]}\n{done.stderr[-3000:]}"
    )
    answers = [line for line in done.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    assert answers, f"heat_worker {mode} printed no result:\n{done.stdout[-2000:]}"
    return dict(json.loads(answers[-1][len(RESULT_PREFIX) :]))


def cell_centres(region_dir: Path) -> Any:
    """The cell centres of ``region_dir``'s mesh, the mean of their face centres (exact on a box).

    From ``constant/polyMesh``: the NeoN mesh leaves its cell centres empty.
    """
    mesh = region_dir / "constant" / "polyMesh"
    points = np.asarray(FoamFile(mesh / "points")[None], dtype=float)
    faces = FoamFile(mesh / "faces")[None]
    owner = np.asarray(FoamFile(mesh / "owner")[None])
    neighbour = np.asarray(FoamFile(mesh / "neighbour")[None])
    face_centres = np.array([points[np.asarray(face)].mean(axis=0) for face in faces])
    cells = np.concatenate([owner, neighbour])
    centres = np.concatenate([face_centres, face_centres[: len(neighbour)]])
    total = np.zeros((cells.max() + 1, 3))
    np.add.at(total, cells, centres)
    return total / np.bincount(cells)[:, None]
