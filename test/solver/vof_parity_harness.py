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


# Shared NeoN VoF-state builder injected into the NeoN driver strings — the
# minimal runtime + damBreak field construction the primitive-parity drivers
# need (rt / alpha1 / phi / p_rgh / U / phase). Replaces the former
# ``NeoInterFoam(...).setup()`` scaffold now that the imperative solver is gone;
# the framework solver ``incompressibleVoFNeon`` builds the same state through
# its ModelSpec ``@build`` steps (create_fields.create_neon_runtime registers
# the identical VoF schemes). ``setup()`` returns a SimpleNamespace holding the
# argList + Time too, so ``Foam::Time``'s raw argList reference stays alive for
# the whole driver (dropping it corrupts later dictionary reads → nan).
NEON_VOF_STATE = r"""
import types as _types
import pybFoam as _pyf
import neon._neon as _nn
import neofoam.neofoam_bindings as _nfb
from neofoam.solver.neoPimpleFoam import _ensure_neon_initialized as _ensure_neon


def setup(argv=None):
    argv = argv or ["vofparity"]
    _ensure_neon(argv)
    _al = _pyf.argList(argv)
    _t = _pyf.Time(_al)
    rt = _nfb.create_adapter_run_time(_t)
    rt.fv_schemes_dict = _nfb.map_fv_schemes(rt.fv_schemes_dict)
    _div = rt.fv_schemes_dict.subDict("divSchemes")
    _div.insert_token_list("div(rhoPhi,U)", _nn.TokenList(["Gauss", "linear"]))
    _div.insert_token_list(
        "div((nuEff*dev2(T(grad(U)))))", _nn.TokenList(["Gauss", "linear"])
    )
    _lap = rt.fv_schemes_dict.subDict("laplacianSchemes")
    for _k in ("laplacian(muf,U)", "laplacian(rAUf,p_rgh)"):
        _lap.insert_token_list(_k, _nn.TokenList(["Gauss", "linear", "uncorrected"]))
    alpha1 = _nfb.read_scalar_volume_field(rt, "alpha.water")
    U = _nfb.read_vector_volume_field(rt, "U")
    p_rgh = _nfb.read_scalar_volume_field(rt, "p_rgh")
    phi = _nfb.create_phi(rt, "U")
    phase = _nfb.read_two_phase_transport_properties(rt)
    return _types.SimpleNamespace(
        rt=rt, alpha1=alpha1, U=U, p_rgh=p_rgh, phi=phi, phase=phase,
        _arg_list=_al, _foam_time=_t,
    )
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


def neon_executors() -> list[str]:
    """NeoN executors available on this build, as ``create_adapter_run_time`` names.

    Always includes ``"Serial"``. Adds ``"CPU"`` when the Kokkos host-parallel
    backend is compiled in (``__has_cpu__``) and ``"GPU"`` when a device is
    actually usable (``gpu_available()`` — a runtime probe; the ``__has_gpu__``
    compile flag is unreliable). Tests parametrize over this so every NeoN
    primitive is verified on each real executor, not just the default Serial.
    """
    import neon._neon as nn

    execs = ["Serial"]
    if getattr(nn, "__has_cpu__", False):
        execs.append("CPU")
    gpu_probe = getattr(nn, "gpu_available", None)
    if callable(gpu_probe) and gpu_probe():
        execs.append("GPU")
    return execs


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
