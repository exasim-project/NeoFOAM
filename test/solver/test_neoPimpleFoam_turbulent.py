# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end turbulent (SA-DDES) run test for the NeoN-based ``neoPimpleFoam``.

Drives the full ``examples/neoPimpleFoam/neoPimpleFoam.cpp`` turbulence path
through the Python bindings: ``TurbulenceModel::create`` selects the LES
``SpalartAllmarasDDES`` model from ``constant/turbulenceProperties``, and each
time step solves the momentum + pressure system *and* the SA ``nuTilda``
transport PDE, then updates ``nut``.

The case is the ready-to-run SA-DDES setup committed under ``test/setup_saddes``
(a seeded pitzDaily); the ``casebuild`` pipeline copies it and meshes it — the
solver settings live in the case dicts, not in this file.

This asserts the binding correctly runs the model end-to-end and that turbulence
actually develops (``nut`` grows from its seed). It deliberately does NOT assert
field-by-field parity against OpenFOAM: the underlying NeoFOAM SA-DDES is already
verified equal to OpenFOAM **per step to 1e-10** in ``test/spalartAllmarasDDES.cpp``,
and instantaneous field parity on a chaotic high-Re case is not a meaningful
metric (two independent solver stacks drift even for laminar flow — see the
``pitzDaily`` note in the port memory).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from neofoam.tooling.casebuild import block_mesh, from_template


def test_neoPimpleFoam_SA_DDES_runs_and_develops_turbulence(tmp_path: Path) -> None:
    """The SA-DDES turbulence path runs end-to-end and nut grows from its seed."""
    repo_root = Path(__file__).parents[2]
    source_case = repo_root / "test" / "setup_saddes"
    case = (from_template(source_case) | block_mesh()).build_at(tmp_path / "sa_ddes")

    # Run in an isolated subprocess: NeoN/Kokkos + OpenFOAM hold per-process global
    # state that does not survive a second in-process solver run. FOAM_SIGFPE is
    # disabled so the signal handler does not abort the wild early SA transient.
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.neoPimpleFoam import NeoPimpleFoam;"
            " NeoPimpleFoam(['neoPimpleFoam']).run()",
        ],
        cwd=str(case.path),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"neoPimpleFoam (SA-DDES) failed (rc={result.returncode}):\n{result.stderr[-3000:]}"
    )

    nut = case.read_field("nut")
    nutilda = case.read_field("nuTilda")

    # Turbulence developed: nut grew from its uniform-0 start to a finite field.
    assert np.all(np.isfinite(nut)), "nut contains non-finite values"
    assert np.all(np.isfinite(nutilda)), "nuTilda contains non-finite values"
    assert float(np.max(nut)) > 1e-7, (
        f"nut did not develop (max nut = {float(np.max(nut)):.3e}); "
        "the SA-DDES model produced no turbulent viscosity"
    )
    print(
        f"SA-DDES developed: max nut = {float(np.max(nut)):.3e}, "
        f"max nuTilda = {float(np.max(nutilda)):.3e} (case {case.path.name})"
    )
