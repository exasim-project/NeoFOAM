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

HERE = Path(__file__).parent
CASES = HERE / "cases"
WORKER = HERE / "heat_worker.py"
RESULT_PREFIX = "#RESULT "


def mesh_copy(case: str, dest: Path, *overlays: str) -> Path:
    """``cases/<case>`` copied to ``dest``, each ``cases/<overlay>`` laid over it, blockMeshed."""
    shutil.copytree(CASES / case, dest)
    for overlay in overlays:
        shutil.copytree(CASES / overlay, dest, dirs_exist_ok=True)
    for blockmesh_dict in sorted(dest.glob("**/system/blockMeshDict")):
        region = blockmesh_dict.parent.parent
        subprocess.run(
            ["blockMesh", "-case", str(region)], check=True, capture_output=True, timeout=120
        )
    return dest


def run_worker(case_dir: Path, mode: str) -> subprocess.CompletedProcess[str]:
    """Run the worker on ``case_dir``, FOAM_SIGFPE off as for the other NeoN solver tests."""
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
