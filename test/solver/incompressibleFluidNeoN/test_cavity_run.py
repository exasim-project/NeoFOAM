# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end cavity run test for the framework ``incompressibleFluidNeoN`` solver.

Builds the lid-driven-cavity case (``test/setup_pimple``) with the ``casebuild``
pipeline — short-transient ``controlDict`` timings + ``blockMesh`` — then runs the
framework solver in an isolated subprocess (NeoN/Kokkos + OpenFOAM keep per-process
global state that does not survive a second in-process run), and asserts the run
completed with finite ``p`` / ``U`` output fields.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from neofoam.tooling.casebuild import block_mesh, from_template, patch

# Short transient: fixed steps (deltaT 0.005) over 100 steps exercises the full
# outer-PIMPLE / inner-PISO machinery while staying fast.
END_TIME = 0.5
N_STEPS = 100  # END_TIME / deltaT


def test_incompressibleFluidNeoN_cavity_runs(tmp_path: Path) -> None:
    """The framework NeoN solver completes the cavity case with finite fields."""
    repo_root = Path(__file__).parents[3]
    source_case = repo_root / "test" / "setup_pimple"
    case = (
        from_template(source_case)
        | patch("system/controlDict", endTime=END_TIME, writeInterval=N_STEPS)
        | block_mesh()
    ).build_at(tmp_path / "cavity")

    # Isolated subprocess: NeoN/Kokkos + OpenFOAM per-process state. FOAM_SIGFPE
    # is disabled so the signal handler does not abort on a benign denormal.
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.incompressibleFluidNeoN import run;"
            " run(['incompressibleFluidNeoN'])",
        ],
        cwd=str(case.path),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"incompressibleFluidNeoN failed (rc={result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    )
    assert "End" in result.stdout

    # ``read_field`` at the endTime dir doubles as the assertion that the run
    # reached END_TIME (it raises if that time directory is absent).
    p = case.read_field("p", time=str(END_TIME))
    u = case.read_field("U", time=str(END_TIME))
    assert np.all(np.isfinite(p)), "p contains non-finite values"
    assert np.all(np.isfinite(u)), "U contains non-finite values"
    assert float(np.max(np.abs(u))) > 0.0, "U stayed identically zero"
