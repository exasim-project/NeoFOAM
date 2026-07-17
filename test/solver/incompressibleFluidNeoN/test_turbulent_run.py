# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""SA-DDES run test for the framework ``incompressibleFluidNeoN`` solver.

Mirror of ``test/solver/test_neoPimpleFoam_turbulent.py`` for the framework
port: the committed seeded-pitzDaily case (``test/setup_saddes``) selects the
LES SpalartAllmarasDDES model via ``constant/turbulenceProperties``; each step
solves momentum + pressure *and* the SA ``nuTilda`` transport, and the NeoN
write hook persists ``nut`` / ``nuTilda`` through ``turb.write``.

The case is built with the ``casebuild`` pipeline (``from_template`` + ``blockMesh``)
and run in an isolated subprocess (NeoN/Kokkos + OpenFOAM keep per-process global
state that does not survive a second in-process run). The solver settings live in
the case dicts, not in this file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neofoam.tooling.casebuild import block_mesh, from_template


@pytest.mark.xfail(
    reason="SpalartAllmarasDDES (LES) has no native NeoN closure and the case uses "
    "wall functions; the turbulence-family merge removed the NeoN C++ factory branch "
    "(design Q-A: require native coverage). Native LES/wall-function coverage lands "
    "from develop into the stack shortly; un-xfail then.",
    strict=False,
)
def test_incompressibleFluidNeoN_SA_DDES_runs(tmp_path: Path) -> None:
    """The SA-DDES turbulence path runs end-to-end and nut/nuTilda land on disk."""
    repo_root = Path(__file__).parents[3]
    source_case = repo_root / "test" / "setup_saddes"
    case = (from_template(source_case) | block_mesh()).build_at(tmp_path / "sa_ddes")

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
        f"incompressibleFluidNeoN (SA-DDES) failed (rc={result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    )

    nut = case.read_field("nut")
    nutilda = case.read_field("nuTilda")

    assert np.all(np.isfinite(nut)), "nut contains non-finite values"
    assert np.all(np.isfinite(nutilda)), "nuTilda contains non-finite values"
    assert float(np.max(nut)) > 1e-7, (
        f"nut did not develop (max nut = {float(np.max(nut)):.3e}); "
        "the SA-DDES model produced no turbulent viscosity"
    )
