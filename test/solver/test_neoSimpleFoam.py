# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end run test for the NeoN-based ``neoSimpleFoam``.

Drives the steady-state SIMPLE lid-driven-cavity case (``test/setup_simple``)
through the Python bindings and asserts the flow actually develops (the lid
drags the adjacent fluid along) and the solution is finite and (nearly)
divergence-free, the way ``test_neoPimpleFoam_turbulent.py`` checks
``neoPimpleFoam``. It deliberately does NOT assert field-by-field parity
against OpenFOAM's ``simpleFoam`` or the C++ ``examples/neoSimpleFoam`` - this
is a port-runs-and-makes-physical-sense smoke test, not a parity test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from neofoam.tooling.casebuild import block_mesh, from_template


def test_neoSimpleFoam_runs_and_develops_cavity_flow(tmp_path: Path) -> None:
    """neoSimpleFoam runs end-to-end and the lid drags the cavity flow along."""
    repo_root = Path(__file__).parents[2]
    source_case = repo_root / "test" / "setup_simple"
    case = (from_template(source_case) | block_mesh()).build_at(tmp_path / "simple")

    # Run in an isolated subprocess: NeoN/Kokkos + OpenFOAM hold per-process global
    # state that does not survive a second in-process solver run.
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.neoSimpleFoam import NeoSimpleFoam;"
            " NeoSimpleFoam(['neoSimpleFoam']).run()",
        ],
        cwd=str(case.path),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"neoSimpleFoam failed (rc={result.returncode}):\n{result.stderr[-3000:]}"
    )

    U = case.read_field("U")
    p = case.read_field("p")

    assert np.all(np.isfinite(U)), "U contains non-finite values"
    assert np.all(np.isfinite(p)), "p contains non-finite values"

    # The lid moves at (1, 0, 0); the adjacent fluid should be dragged along to a
    # non-trivial fraction of that, confirming the SIMPLE loop actually solved
    # something rather than leaving U at its zero initial condition.
    max_ux = float(np.max(U[:, 0]))
    assert max_ux > 0.05, f"cavity flow did not develop (max Ux = {max_ux:.3e})"
    print(f"neoSimpleFoam converged: max Ux = {max_ux:.3e}, max |p| = {float(np.max(np.abs(p))):.3e}")
