# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The heated plate case, its probe values, and the fluid's written interface ``T``."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

HERE = Path(__file__).parent
CASES = HERE / "cases"
CASE = CASES / "heatedPlate"
RESULT_PREFIX = "#RESULT "
TIMEOUT = 600

#: the probe at 0.03 s (to 1e-6 K): passes per time step, the fluid's wall T (mean, min, max)
PROBE_PASSES = [15, 16, 15]
PROBE_WALL_T = (301.315643, 300.962216, 301.347602)


def mesh_copy(dest: Path, overlay: Optional[str] = None, *, tools: tuple[str, ...]) -> Path:
    """``cases/heatedPlate`` in ``dest``, ``cases/<overlay>`` over it, meshed by ``tools``."""
    shutil.copytree(CASE, dest)
    if overlay is not None:
        shutil.copytree(CASES / overlay, dest, dirs_exist_ok=True)
    for region in ("fluid", "solid"):
        for tool in tools:
            subprocess.run(
                [tool, "-case", region], cwd=dest, check=True, capture_output=True, timeout=TIMEOUT
            )
    return dest


def run(command: list[str], case: Path) -> str:
    """The stdout of ``command`` run in ``case``; it must succeed."""
    done = subprocess.run(
        command,
        cwd=case,
        env={**os.environ, "FOAM_SIGFPE": "false", "OMPI_MCA_rmaps_base_oversubscribe": "1"},
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    assert done.returncode == 0, (
        f"{command} failed (rc={done.returncode}):\n{done.stdout[-2000:]}\n{done.stderr[-3000:]}"
    )
    return done.stdout


def result(command: list[str], case: Path) -> dict[str, Any]:
    """What ``command`` printed on its ``#RESULT`` line, run in ``case``."""
    stdout = run(command, case)
    answers = [line for line in stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    assert answers, f"{command} printed no result:\n{stdout[-2000:]}"
    return dict(json.loads(answers[-1][len(RESULT_PREFIX) :]))


def written_interface_T(fluid: Path, part: Path, staged: Path) -> Any:
    """The interface values of ``T`` at 0.03 s as written in ``part`` of the region ``fluid``."""
    worker = [sys.executable, str(HERE / "_written_T_worker.py")]
    args = [str(fluid / "system"), str(part), "0.03", "interface", str(staged)]
    return np.asarray(result([*worker, *args], fluid)["T"])
