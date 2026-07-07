# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Reusable pybFoam ``fvc::`` reference-parity harness for the NeoN VoF solver.

pybFoam (OpenFOAM ``Foam::Time``) and NeoN/nfb both embed OpenFOAM per-process
global state and MUST NOT coexist in one interpreter (a second ``Time`` corrupts
the first — reads come back nan). This harness therefore drives a **pyf reference
subprocess** and a **NeoN subprocess** separately, exchanging arrays through
``.npy`` files in a shared damBreak case directory:

    metrics = run_parity(case, PYF_REF_DRIVER, NEON_DRIVER)

``run_parity`` first runs the pyf driver (which reads the OpenFOAM reference
field on the damBreak mesh and writes ``*.npy`` references + shared inputs), then
the NeoN driver (which loads those arrays, computes the same quantity in NeoN and
prints ``KEY value`` metric lines). Both drivers run with ``cwd == case``.

Face-field parity assumes the nfb<->OpenFOAM adapter preserves internal-face
ordering; the ``mag_sf`` client validates that assumption face-by-face — read it
together with the ``sn_grad`` client (both are face fields with the same ordering
contract).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("FOAM_SIGFPE", "false")


# Shared relative-error helper — injected into both driver strings so the pyf and
# NeoN sides use the identical definition: rel = max|a-b| / (max|b| + tiny).
REL_ERR = r"""
def rel_err(a, b):
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-300))
"""


def prepare_case(dest: Path) -> None:
    """Copy the damBreak tutorial and run blockMesh + setFields into ``dest``."""
    repo_root = Path(__file__).parent.parent.parent
    src = repo_root / "tutorials" / "damBreak"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    zero = dest / "0"
    if not zero.exists():
        shutil.copytree(dest / "0.orig", zero)
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    for cmd in (["blockMesh"], ["setFields"]):
        r = subprocess.run(
            cmd, cwd=str(dest), env=env, capture_output=True, text=True, timeout=180
        )
        assert r.returncode == 0, f"{cmd[0]} failed:\n{r.stderr[-2000:]}"


def _run_subprocess(case: Path, driver: str, timeout: int = 300) -> str:
    """Run one driver string as ``python -c`` in ``case``; return its stdout."""
    env = {**os.environ, "FOAM_SIGFPE": "false"}
    r = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=str(case),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert r.returncode == 0, f"driver failed:\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}"
    assert "END_OK" in r.stdout, f"driver did not finish cleanly:\n{r.stdout[-2000:]}"
    return r.stdout


def _parse_metrics(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def run_parity(
    case: Path, pyf_driver: str, neon_driver: str, timeout: int = 300
) -> dict[str, float]:
    """Run the pyf reference subprocess then the NeoN subprocess in ``case``.

    The pyf driver writes ``.npy`` references into ``case``; the NeoN driver reads
    them back and emits ``KEY value`` metric lines. Metrics from both stdout
    streams are merged and returned.
    """
    pyf_out = _run_subprocess(case, pyf_driver, timeout)
    neon_out = _run_subprocess(case, neon_driver, timeout)
    metrics = _parse_metrics(pyf_out)
    metrics.update(_parse_metrics(neon_out))
    return metrics
