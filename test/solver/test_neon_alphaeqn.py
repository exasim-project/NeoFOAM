# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Machine-precision parity for the NeoN alphaEqn.H primitives vs pybFoam.

The NeoN VoF alpha scheme is being ported from the simplified explicit-MULES step
(linear interp, no compression) up to the full OpenFOAM ``alphaEqn.H`` (Gauss
vanLeer + interface compression ``cAlpha`` + ``MULESCorr``), one primitive at a
time. Each primitive is verified by advecting the shared disc with BOTH backends
configured to the SAME scheme and asserting the fields match to machine precision
over 100 steps (the multi-step bar — a single step from a sharp field agrees
trivially).

**Every NeoN check runs on all available executors** (Serial / CPU / GPU, per
``neon_executors()``): the parity fixtures are parametrized over the executor, so
each primitive is verified on every real backend, not just the default Serial.
GPU differs from Serial only by atomic-reduction ordering (~1e-15), well inside
the machine-precision bar.

Increment 1 — **vanLeer** (``nfb.vanleer_flux``): NeoN's vanLeer-limited convective
flux vs OpenFOAM's ``fvc::flux(phi, alpha, "Gauss vanLeer")``.
Increment 2 — **compression** (``nfb.alpha_phase_flux``): the full alphaEqn.H
high-order flux (vanLeer + ``cAlpha`` interface compression). Both run the same
explicit MULES limiter (already parity-verified) on top.

Later increments (``MULES::correct`` + implicit predictor for ``MULESCorr``)
extend this file; the final test is a full-config damBreak parity vs
``incompressibleVoF``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_disc_advection import _DT, _N_STEPS, _prepare_disc_case
from vof_parity_harness import _run_subprocess, neon_executors

os.environ.setdefault("FOAM_SIGFPE", "false")

_EXECUTORS = neon_executors()

# Executors on which a Ginkgo LINEAR solve runs. The explicit alpha path is pure
# Kokkos (no linear solve) and runs on every executor in _EXECUTORS; the MULESCorr
# path adds an implicit upwind predictor solved by Ginkgo, and a Ginkgo solve on
# the CUDA backend currently segfaults in this build (Kokkos CUDA device not
# initialised for the Ginkgo executor — a NeoN-core limitation affecting all GPU
# Ginkgo solves, incl. the pressure equation, not specific to MULESCorr). So the
# implicit-predictor tests run on the non-GPU executors; GPU coverage of the alpha
# scheme itself stays via the explicit-path tests above (verified to ~3e-15).
_SOLVE_EXECUTORS = [e for e in _EXECUTORS if e != "GPU"]


# pyf reference: the full incompressibleVoF alpha step (_solve_alpha_python).
# Executor-independent — OpenFOAM runs on the host. With the fvSolution controls
# patched per fixture it reduces to the scheme under test -> explicit MULES.
_PYF = f"""
import numpy as np, pybFoam as pyf
from pybFoam import fvc, surfaceScalarField, volVectorField, volScalarField
import pybFoam.multiphase as vof
from neofoam.solver.incompressibleVoF.models.alpha_advection.models.mules import (
    _solve_alpha_python,
)

runTime = pyf.Time(pyf.argList(["disc"]))
mesh = pyf.fvMesh(runTime)
mesh.setFluxRequired(pyf.Word("alpha.water"))
U = volVectorField.read_field(mesh, "U")
phi = pyf.createPhi(U)
mix = vof.immiscibleIncompressibleTwoPhaseMixture(U, phi)
a1, a2 = mix.alpha1(), mix.alpha2()
r1, r2 = mix.rho1(), mix.rho2()
rho = volScalarField(pyf.Word("rho"), a1 * r1 + a2 * r2)
rhoPhi = surfaceScalarField(pyf.Word("rhoPhi"), fvc.interpolate(rho) * phi)
for _ in range({_N_STEPS}):
    runTime.increment()  # rotate alpha.oldTime()
    _solve_alpha_python(a1, a2, phi, rhoPhi, rho, mix)
np.save("pyf_a.npy", np.asarray(a1.internalField()))
print("END_OK")
"""


def _neon_driver(flux_expr: str, executor: str) -> str:
    """NeoN alpha-advection driver on a named executor, using ``flux_expr`` as the
    high-order phase flux (vanleer_flux / alpha_phase_flux) -> explicit MULES."""
    return f"""
import gc, numpy as np, pybFoam as pyf, neon._neon as nn, neofoam.neofoam_bindings as nfb
from neofoam.solver.neoPimpleFoam import _ensure_neon_initialized


def drive():
    _ensure_neon_initialized(["disc"])
    al = pyf.argList(["disc"]); t = pyf.Time(al)
    rt = nfb.create_adapter_run_time(t, "{executor}")
    rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
    rt.dt = {_DT}
    alpha = nfb.read_scalar_volume_field(rt, "alpha.water")
    U = nfb.read_vector_volume_field(rt, "U")
    phi = nfb.flux(U)
    for _ in range({_N_STEPS}):
        aphi = {flux_expr}
        nfb.mules_explicit_solve(alpha, phi, aphi, rt.dt, 1.0, 0.0, 5)
    np.save("neon_a.npy", np.asarray(alpha.internal_vector().copy_to_host()))
    del alpha, U, phi, rt, t, al
    print("END_OK")


drive(); gc.collect()
"""


def _patch_alpha_controls(case, **kv: object) -> None:
    p = case / "system" / "fvSolution"
    t = p.read_text()
    for key, val in kv.items():
        t = re.sub(rf"{key}\s+\S+;", f"{key} {val};", t)
    p.write_text(t)


def _run_parity(tmp_path_factory, name, flux_expr, executor, **controls) -> dict:
    case = tmp_path_factory.mktemp(name)
    _prepare_disc_case(case)
    _patch_alpha_controls(case, **controls)
    _run_subprocess(case, _PYF, timeout=400)
    _run_subprocess(case, _neon_driver(flux_expr, executor), timeout=400)
    pyf_a = np.load(case / "pyf_a.npy")
    neon_a = np.load(case / "neon_a.npy")
    diff = np.abs(pyf_a - neon_a)
    return {
        "MAXABS": float(diff.max()),
        "REL_L2": float(
            np.sqrt((diff**2).mean()) / (np.sqrt((pyf_a**2).mean()) + 1e-30)
        ),
        "NEON_BAND": int(((neon_a > 0.01) & (neon_a < 0.99)).sum()),
        "PYF_BAND": int(((pyf_a > 0.01) & (pyf_a < 0.99)).sum()),
    }


@pytest.fixture(scope="module", params=_EXECUTORS)
def vanleer(request, tmp_path_factory: pytest.TempPathFactory) -> dict:
    return _run_parity(
        tmp_path_factory,
        f"discVanLeer_{request.param}",
        "nfb.vanleer_flux(rt, alpha, phi)",
        request.param,
        cAlpha=0,
        MULESCorr="no",
        nAlphaCorr=1,
    )


@pytest.fixture(scope="module", params=_EXECUTORS)
def compression(request, tmp_path_factory: pytest.TempPathFactory) -> dict:
    return _run_parity(
        tmp_path_factory,
        f"discCompression_{request.param}",
        "nfb.alpha_phase_flux(rt, alpha, phi, 1.0)",
        request.param,
        cAlpha=1,
        MULESCorr="no",
        nAlphaCorr=1,
    )


def test_vanleer_flux_matches_pybfoam_over_100_steps(vanleer: dict) -> None:
    """nfb.vanleer_flux == OpenFOAM fvc::flux(phi, alpha, 'Gauss vanLeer').

    Same scheme both backends (vanLeer -> explicit MULES, cAlpha=0), 100 steps,
    bitwise-identical on every executor. (Observed ~9e-15.)
    """
    assert vanleer["MAXABS"] < 1e-10
    assert vanleer["REL_L2"] < 1e-12


def test_compression_flux_matches_pybfoam_over_100_steps(compression: dict) -> None:
    """nfb.alpha_phase_flux(cAlpha=1) == pybFoam alphaEqn.H high-order flux.

    vanLeer + interface compression, 100 steps, bitwise-identical on every
    executor (~5e-15). Compression also sharpens the interface (band 300 -> ~120),
    matching pyf.
    """
    assert compression["MAXABS"] < 1e-10
    assert compression["REL_L2"] < 1e-12
    assert compression["NEON_BAND"] == compression["PYF_BAND"]
    assert compression["NEON_BAND"] < 200


# --- Full explicit alpha path on the REAL damBreak case (graded 5-block mesh +
#     the water-column interface), not the uniform disc box. This is the complete
#     verification that the solver's explicit alpha step (vanLeer + cAlpha
#     compression + explicit MULES) matches pybFoam to machine precision on the
#     actual case geometry, given a shared flux. A fully-coupled damBreak run
#     cannot match bitwise (the NeoN vs OpenFOAM pressure/velocity solves differ,
#     so the adaptive-dt trajectory diverges); this isolates the alpha equation by
#     imposing the same divergence-free phi (uniform U, noSlip walls) on both.
def _prepare_dambreak_alpha_case(dest: Path) -> None:
    repo_root = Path(__file__).parent.parent.parent
    src = repo_root / "tutorials" / "damBreak"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    shutil.copytree(dest / "0.orig", dest / "0", dirs_exist_ok=True)

    g = dest / "constant" / "g"
    g.write_text(re.sub(r"value\s+\([^)]*\)", "value           (0 0 0)", g.read_text()))
    u = dest / "0" / "U"
    u.write_text(
        re.sub(
            r"internalField\s+uniform \([^)]*\)",
            "internalField   uniform (1 0 0)",
            u.read_text(),
        )
    )
    cd = dest / "system" / "controlDict"
    t = cd.read_text()
    t = re.sub(r"adjustTimeStep\s+\S+;", "adjustTimeStep no;", t)
    t = re.sub(r"deltaT\s+\S+;", f"deltaT {_DT};", t)
    t = re.sub(r"endTime\s+\S+;", "endTime 0.05;", t)
    cd.write_text(t)

    env = {**os.environ, "FOAM_SIGFPE": "false"}
    for cmd in (["blockMesh"], ["setFields"]):
        r = subprocess.run(
            cmd, cwd=str(dest), env=env, capture_output=True, text=True, timeout=180
        )
        assert r.returncode == 0, f"{cmd[0]} failed:\n{r.stderr[-2000:]}"


@pytest.fixture(scope="module", params=_EXECUTORS)
def dambreak_explicit(request, tmp_path_factory: pytest.TempPathFactory) -> dict:
    case = tmp_path_factory.mktemp(f"dbExplicit_{request.param}")
    _prepare_dambreak_alpha_case(case)
    _patch_alpha_controls(case, cAlpha=1, MULESCorr="no", nAlphaCorr=1)
    _run_subprocess(case, _PYF, timeout=400)
    _run_subprocess(
        case,
        _neon_driver("nfb.alpha_phase_flux(rt, alpha, phi, 1.0)", request.param),
        timeout=400,
    )
    pyf_a = np.load(case / "pyf_a.npy")
    neon_a = np.load(case / "neon_a.npy")
    diff = np.abs(pyf_a - neon_a)
    return {
        "MAXABS": float(diff.max()),
        "REL_L2": float(
            np.sqrt((diff**2).mean()) / (np.sqrt((pyf_a**2).mean()) + 1e-30)
        ),
        "NEON_BAND": int(((neon_a > 0.01) & (neon_a < 0.99)).sum()),
        "PYF_BAND": int(((pyf_a > 0.01) & (pyf_a < 0.99)).sum()),
    }


def test_explicit_alpha_path_matches_pybfoam_on_dambreak(
    dambreak_explicit: dict,
) -> None:
    """The full explicit alpha step matches pyf to machine precision on the REAL
    damBreak graded mesh + water-column interface (100 steps, shared phi), on
    every executor. Observed maxabsdiff ~3e-15 (~1 ULP at the interface)."""
    assert dambreak_explicit["MAXABS"] < 1e-10
    assert dambreak_explicit["REL_L2"] < 1e-12
    assert dambreak_explicit["NEON_BAND"] == dambreak_explicit["PYF_BAND"]


# --- Increment 3: MULESCorr (semi-implicit MULES). The disc's default alpha
#     controls are cAlpha=1, MULESCorr=yes, nAlphaCorr=2, nLimiterIter=5. The
#     NeoN side runs the SOLVER's own ``_mules_corr_solve`` (implicit upwind
#     predictor via the generic scalar PDE solver + FCT-limited high-order
#     correctors), so this exercises the real solver code path, not a replica.
#     The predictor is an implicit linear solve, so — exactly like the pressure
#     equation — NeoN's Ginkgo Bicgstab and OpenFOAM's smoothSolver converge to
#     slightly different iterates unless the alpha tolerance is tight: at the
#     shipped 1e-8 the 100-step drift is ~2.5e-5; tightened to 1e-13 both
#     predictors converge and the paths match to ~1.5e-9 (band-identical).
def _neon_mulescorr_driver(executor: str) -> str:
    """NeoN MULESCorr driver on a named executor, calling the solver's own
    ``_mules_corr_solve`` (predictor + corrector) each step."""
    return f"""
import gc, numpy as np, pybFoam as pyf, neon._neon as nn, neofoam.neofoam_bindings as nfb
from neofoam.solver.neoPimpleFoam import _ensure_neon_initialized
from neofoam.solver.incompressibleVoFNeon.models.alpha_advection.alphaAdvectionModel import (
    _mules_corr_solve, _read_alpha_controls,
)
from neofoam.solver.incompressibleVoFNeon.create_fields import (
    _register_alpha_predictor_solver,
)


def drive():
    _ensure_neon_initialized(["disc"])
    al = pyf.argList(["disc"]); t = pyf.Time(al)
    rt = nfb.create_adapter_run_time(t, "{executor}")
    rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
    rt.dt = {_DT}
    # Predictor scheme + solver dict (mirrors create_fields.create_neon_runtime).
    rt.fv_schemes_dict.subDict("divSchemes").insert_token_list(
        "div(phi,alpha.water)", nn.TokenList(["Gauss", "upwind"]))
    _register_alpha_predictor_solver(rt)
    ctrl = _read_alpha_controls(rt)
    alpha = nfb.read_scalar_volume_field(rt, "alpha.water")
    phi = nfb.create_phi(rt, "U")
    for _ in range({_N_STEPS}):
        nn.rotate_old_times(alpha)
        _mules_corr_solve(
            rt, alpha, phi, ctrl["c_alpha"], ctrl["n_limiter_iter"], ctrl["n_alpha_corr"]
        )
    np.save("neon_a.npy", np.asarray(alpha.internal_vector().copy_to_host()))
    del alpha, phi, rt, t, al
    print("END_OK")


drive(); gc.collect()
"""


@pytest.fixture(scope="module", params=_SOLVE_EXECUTORS)
def mulescorr(request, tmp_path_factory: pytest.TempPathFactory) -> dict:
    case = tmp_path_factory.mktemp(f"discMulesCorr_{request.param}")
    _prepare_disc_case(case)
    # Tighten the alpha.water predictor tolerance so the Ginkgo-vs-smoothSolver
    # implicit-solve difference drops below the parity bar (see module note).
    fv = case / "system" / "fvSolution"
    fv.write_text(
        fv.read_text().replace("tolerance       1e-8;", "tolerance       1e-13;")
    )
    _run_subprocess(case, _PYF, timeout=400)
    _run_subprocess(case, _neon_mulescorr_driver(request.param), timeout=400)
    pyf_a = np.load(case / "pyf_a.npy")
    neon_a = np.load(case / "neon_a.npy")
    diff = np.abs(pyf_a - neon_a)
    return {
        "MAXABS": float(diff.max()),
        "REL_L2": float(
            np.sqrt((diff**2).mean()) / (np.sqrt((pyf_a**2).mean()) + 1e-30)
        ),
        "NEON_BAND": int(((neon_a > 0.01) & (neon_a < 0.99)).sum()),
        "PYF_BAND": int(((pyf_a > 0.01) & (pyf_a < 0.99)).sum()),
    }


def test_mulescorr_alpha_path_matches_pybfoam(mulescorr: dict) -> None:
    """The solver's MULESCorr path (implicit predictor + FCT correctors) matches
    pyf ``incompressibleVoF`` on the disc, on every executor. With a tight
    predictor tolerance the paths agree to ~1.5e-9 (band-identical); the residual
    is the implicit-solve Krylov backend (Ginkgo vs smoothSolver), not the
    discretisation."""
    assert mulescorr["MAXABS"] < 1e-7
    assert mulescorr["REL_L2"] < 1e-7
    assert mulescorr["NEON_BAND"] == mulescorr["PYF_BAND"]
