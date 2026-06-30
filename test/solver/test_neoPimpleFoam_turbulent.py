# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end turbulent (SA-DDES) run test for the NeoN-based ``neoPimpleFoam``.

Drives the full ``examples/neoPimpleFoam/neoPimpleFoam.cpp`` turbulence path
through the Python bindings: ``TurbulenceModel::create`` selects the LES
``SpalartAllmarasDDES`` model from ``constant/turbulenceProperties``, and each
time step solves the momentum + pressure system *and* the SA ``nuTilda``
transport PDE, then updates ``nut``.

The case is the ready-to-run SA-DDES setup committed under ``test/setup_saddes``
(a seeded pitzDaily); the test just copies it, meshes it and runs it — the
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
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _prepare_case(source: Path, dest: Path) -> None:
    """Copy the committed SA-DDES case and generate its mesh."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    result = subprocess.run(
        ["blockMesh", "-case", str(dest)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"blockMesh failed: {result.stderr}"


def _final_time_dir(case: Path) -> Path:
    times = sorted(
        (
            d
            for d in case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and float(d.name) > 0
        ),
        key=lambda d: float(d.name),
    )
    assert times, f"no output time directories in {case}"
    return times[-1]


def _read_internal(time_dir: Path, field_name: str) -> np.ndarray:
    """Parse an OpenFOAM scalar field's internalField (uniform or nonuniform)."""
    txt = (time_dir / field_name).read_text()
    m = re.search(r"internalField\s+nonuniform[^(]*\(([^)]*)\)", txt, re.S)
    if m:
        return np.array([float(x) for x in m.group(1).split()])
    m = re.search(r"internalField\s+uniform\s+([-0-9.eE+]+)", txt)
    assert m, f"could not parse internalField of {field_name}"
    return np.array([float(m.group(1))])


def test_neoPimpleFoam_SA_DDES_runs_and_develops_turbulence(tmp_path: Path) -> None:
    """The SA-DDES turbulence path runs end-to-end and nut grows from its seed."""
    repo_root = Path(__file__).parent.parent.parent
    source = repo_root / "test" / "setup_saddes"
    case = tmp_path / "sa_ddes"
    _prepare_case(source, case)

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
        cwd=str(case),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"neoPimpleFoam (SA-DDES) failed (rc={result.returncode}):\n{result.stderr[-3000:]}"
    )

    final = _final_time_dir(case)
    nut = _read_internal(final, "nut")
    nutilda = _read_internal(final, "nuTilda")

    # Turbulence developed: nut grew from its uniform-0 start to a finite field.
    assert np.all(np.isfinite(nut)), "nut contains non-finite values"
    assert np.all(np.isfinite(nutilda)), "nuTilda contains non-finite values"
    assert float(np.max(nut)) > 1e-7, (
        f"nut did not develop (max nut = {float(np.max(nut)):.3e}); "
        "the SA-DDES model produced no turbulent viscosity"
    )
    print(
        f"SA-DDES developed: max nut = {float(np.max(nut)):.3e}, "
        f"max nuTilda = {float(np.max(nutilda)):.3e} (case {case.name})"
    )
