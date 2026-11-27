# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end run test for the NeoN-based ``neorhoPimpleFoam``.

Drives the transient compressible PIMPLE duct case (``test/setup_rhoPimple``)
through the Python bindings and asserts the flow is finite, develops in the
expected direction, and - the key correctness check - that rho == p/(R*T)
holds in the written fields, confirming the perfectGas thermo-update wiring
actually ran. It deliberately does NOT assert field-by-field parity against
OpenFOAM's ``rhoPimpleFoam`` - this is a port-runs-and-makes-physical-sense
smoke test, not a parity test, the same way ``test_neoSimpleFoam.py`` treats
``neoSimpleFoam``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from neofoam.tooling.casebuild import block_mesh, from_template

UNIVERSAL_GAS_CONSTANT = 8314.46261815324  # J/(kmol K)
MOL_WEIGHT = 28.9  # matches test/setup_rhoPimple/constant/thermophysicalProperties


def test_neorhoPimpleFoam_runs_and_develops_duct_flow(tmp_path: Path) -> None:
    """neorhoPimpleFoam runs end-to-end and the EOS stays self-consistent."""
    repo_root = Path(__file__).parents[2]
    source_case = repo_root / "test" / "setup_rhoPimple"
    case = (from_template(source_case) | block_mesh()).build_at(tmp_path / "rhopimple")

    # Run in an isolated subprocess: NeoN/Kokkos + OpenFOAM hold per-process global
    # state that does not survive a second in-process solver run.
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.neorhoPimpleFoam import NeoRhoPimpleFoam;"
            " NeoRhoPimpleFoam(['neorhoPimpleFoam']).run()",
        ],
        cwd=str(case.path),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"neorhoPimpleFoam failed (rc={result.returncode}):\n{result.stderr[-3000:]}"
    )

    U = case.read_field("U")
    p = case.read_field("p")
    T = case.read_field("T")
    rho = case.read_field("rho")

    assert np.all(np.isfinite(U)), "U contains non-finite values"
    assert np.all(np.isfinite(p)), "p contains non-finite values"
    assert np.all(np.isfinite(T)), "T contains non-finite values"
    assert np.all(np.isfinite(rho)), "rho contains non-finite values"

    # EOS consistency: proves the perfectGas thermo-update wiring (psi.assign/
    # rho.assign each outer iteration) actually ran, independent of how far the
    # flow itself developed.
    R = UNIVERSAL_GAS_CONSTANT / MOL_WEIGHT
    np.testing.assert_allclose(
        rho, p / (R * T), rtol=1e-8, err_msg="rho != p/(R*T); EOS update is not wired correctly"
    )

    # Directional sanity: the duct's inlet drives U, 1 m/s; mean flow should stay
    # close to that rather than collapsing to the zero initial condition.
    mean_ux = float(np.mean(U[:, 0]))
    assert mean_ux > 0.5, f"duct flow did not develop (mean Ux = {mean_ux:.3e})"
    print(
        f"neorhoPimpleFoam ran: mean Ux = {mean_ux:.3e}, "
        f"p range = [{float(np.min(p)):.1f}, {float(np.max(p)):.1f}], "
        f"T range = [{float(np.min(T)):.2f}, {float(np.max(T)):.2f}]"
    )
