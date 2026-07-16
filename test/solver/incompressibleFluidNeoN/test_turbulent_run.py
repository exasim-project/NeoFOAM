# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""SA-DDES run test for the framework ``incompressibleFluidNeoN`` solver.

Mirror of ``test/solver/test_neoPimpleFoam_turbulent.py`` for the framework
port: the committed seeded-pitzDaily case (``test/setup_saddes``) selects the
LES SpalartAllmarasDDES model via ``constant/turbulenceProperties``; each step
solves momentum + pressure *and* the SA ``nuTilda`` transport, and the NeoN
write hook persists ``nut`` / ``nuTilda`` through ``turb.write``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


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


@pytest.mark.xfail(
    reason="SpalartAllmarasDDES (LES) has no native NeoN closure and the case uses "
    "wall functions; the turbulence-family merge removed the NeoN C++ factory branch "
    "(design Q-A: require native coverage). Native LES/wall-function coverage lands "
    "from develop into the stack shortly; un-xfail then.",
    strict=False,
)
def test_incompressibleFluidNeoN_SA_DDES_runs(tmp_path: Path) -> None:
    """The SA-DDES turbulence path runs end-to-end and nut/nuTilda land on disk."""
    repo_root = Path(__file__).parent.parent.parent.parent
    source = repo_root / "test" / "setup_saddes"
    case = tmp_path / "sa_ddes"
    _prepare_case(source, case)

    env = {**os.environ, "FOAM_SIGFPE": "false"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.incompressibleFluidNeoN import run;"
            " run(['incompressibleFluidNeoN'])",
        ],
        cwd=str(case),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"incompressibleFluidNeoN (SA-DDES) failed (rc={result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    )

    final = _final_time_dir(case)
    nut = _read_internal(final, "nut")
    nutilda = _read_internal(final, "nuTilda")

    assert np.all(np.isfinite(nut)), "nut contains non-finite values"
    assert np.all(np.isfinite(nutilda)), "nuTilda contains non-finite values"
    assert float(np.max(nut)) > 1e-7, (
        f"nut did not develop (max nut = {float(np.max(nut)):.3e}); "
        "the SA-DDES model produced no turbulent viscosity"
    )
