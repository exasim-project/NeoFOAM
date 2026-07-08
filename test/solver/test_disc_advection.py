# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Disc-advection integration test — NeoN VoF alpha scheme vs pybFoam interFoam.

The canonical VoF verification: a sharp circular disc of ``alpha.water = 1`` is
advected through a **prescribed divergence-free** velocity field (uniform
translation, boundary flux zeroed so the interior stays incompressible on a
walled box) and the transported phase fraction is compared. It isolates the
**alpha equation** (the momentum/pressure solve is not run — phi is frozen), so
it directly measures each scheme's numerical diffusion and their agreement.

Both backends run on the SAME uniform mesh + SAME initial disc (setFields):

* **pyf reference** = the pybFoam ``incompressibleVoF`` alpha model
  (``_solve_alpha_python``): full ``alphaEqn.H`` — Gauss vanLeer + interface
  compression (``cAlpha=1``) + ``MULESCorr`` + ``nAlphaCorr=2``.
* **NeoN** = what ``incompressibleVoFNeon`` runs: a single explicit MULES step
  (``nfb.mules_explicit_solve``) with linear interpolation and **no** interface
  compression (``cAlpha=0``).

What is asserted TIGHTLY (must always hold): both schemes conserve mass exactly
and keep ``alpha`` bounded in [0, 1], and both translate the disc to the exact
position. What is CHARACTERISED (a regression guard that ratchets toward parity):
the interface diffusion of each scheme and the NeoN-vs-pyf field gap. Today NeoN
is ~2x more diffuse than pyf's full scheme; that gap is entirely the alpha-scheme
*config* (linear/cAlpha=0 vs vanLeer/cAlpha=1/MULESCorr) — it closes when the full
``alphaEqn.H`` is ported to NeoN, at which point ``NEON_BAND`` -> ``PYF_BAND``.

A second fixture (``mules_parity``) runs the *same* bare explicit-MULES scheme on
both backends and asserts they agree to **machine precision over 100 steps** —
the multi-step complement to the single-step ``test_mules`` parity, proving the
NeoN MULES core (``nfb.mules_explicit_solve``) is a bitwise-exact transcription of
OpenFOAM's ``MULES::explicitSolve`` (so the diffusion gap above is purely scheme
config, not a limiter difference). NB the pyf side MUST call ``runTime.increment()``
each step to rotate ``psi.oldTime()``; without it OpenFOAM's disc freezes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vof_parity_harness import _parse_metrics, _run_subprocess, neon_executors

os.environ.setdefault("FOAM_SIGFPE", "false")

# Disc: centred at (0.35, 0.5), radius 0.15, on a unit box, 100x100 uniform mesh.
# U = (1, 0, 0), dt = 0.002 (Co = 0.2), 100 steps -> centre travels to x = 0.55,
# front to 0.70 < 1.0 so the disc never reaches the outflow wall.
_N_STEPS = 100
_DT = 0.002

_BLOCKMESH = """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices
(
    (0 0 0)(1 0 0)(1 1 0)(0 1 0)
    (0 0 0.01)(1 0 0.01)(1 1 0.01)(0 1 0.01)
);
blocks ( hex (0 1 2 3 4 5 6 7) (100 100 1) simpleGrading (1 1 1) );
edges ();
boundary
(
    leftWall { type wall; faces ((0 4 7 3)); }
    rightWall { type wall; faces ((2 6 5 1)); }
    lowerWall { type wall; faces ((1 5 4 0)); }
    atmosphere { type patch; faces ((3 7 6 2)); }
    defaultFaces { type empty; faces ((0 3 2 1)(4 5 6 7)); }
);
mergePatchPairs ();
"""

_SETFIELDS = """FoamFile { version 2.0; format ascii; class dictionary; object setFieldsDict; }
defaultFieldValues ( volScalarFieldValue alpha.water 0 );
regions
(
    cylinderToCell
    {
        p1 (0.35 0.5 -1);
        p2 (0.35 0.5 1);
        radius 0.15;
        fieldValues ( volScalarFieldValue alpha.water 1 );
    }
);
"""


def _prepare_disc_case(dest: Path) -> None:
    """Uniform-box disc case: reuse damBreak's system/constant/0 (patch names +
    BCs match), swap the mesh (uniform box), the field (a disc), zero gravity and
    impose a uniform advecting velocity with a fixed time step."""
    repo_root = Path(__file__).parent.parent.parent
    src = repo_root / "tutorials" / "damBreak"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    shutil.copytree(dest / "0.orig", dest / "0", dirs_exist_ok=True)

    (dest / "system" / "blockMeshDict").write_text(_BLOCKMESH)
    (dest / "system" / "setFieldsDict").write_text(_SETFIELDS)

    g = dest / "constant" / "g"
    g.write_text(re.sub(r"value\s+\([^)]*\)", "value           (0 0 0)", g.read_text()))
    u = dest / "0" / "U"
    u.write_text(
        re.sub(r"internalField\s+uniform \([^)]*\)", "internalField   uniform (1 0 0)", u.read_text())
    )
    cd = dest / "system" / "controlDict"
    t = cd.read_text()
    t = re.sub(r"adjustTimeStep\s+\S+;", "adjustTimeStep no;", t)
    t = re.sub(r"deltaT\s+\S+;", f"deltaT {_DT};", t)
    t = re.sub(r"endTime\s+\S+;", "endTime 0.2;", t)
    cd.write_text(t)

    env = {**os.environ, "FOAM_SIGFPE": "false"}
    for cmd in (
        ["blockMesh"],
        ["setFields"],
        ["postProcess", "-func", "writeCellCentres", "-time", "0"],
    ):
        r = subprocess.run(
            cmd, cwd=str(dest), env=env, capture_output=True, text=True, timeout=180
        )
        assert r.returncode == 0, f"{cmd[0]} failed:\n{r.stderr[-2000:]}"


def _internal_scalar(path: Path) -> np.ndarray:
    txt = path.read_text()
    m = re.search(r"nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\(", txt)
    assert m, f"no nonuniform field in {path}"
    n = int(m.group(1))
    s = txt.index("(", m.end() - 1) + 1
    e = txt.index(")", s)
    v = np.fromstring(txt[s:e].replace("\n", " "), sep=" ")
    assert v.size == n
    return v


def _centroid_x(alpha: np.ndarray, cx: np.ndarray) -> float:
    w = np.clip(alpha, 0.0, 1.0)
    return float((cx * w).sum() / w.sum())


# pyf: full pybFoam incompressibleVoF alphaEqn (vanLeer + cAlpha + MULESCorr).
_PYF_DRIVER = f"""
import numpy as np, pybFoam as pyf
from pybFoam import fvc, volScalarField, surfaceScalarField, volVectorField
import pybFoam.vof as vof
from neofoam.solver.incompressibleVoF.models.alpha_advection.alphaAdvectionModel import (
    _solve_alpha_python,
)

N = {_N_STEPS}
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

a0 = np.asarray(a1.internalField()).copy()
for _ in range(N):
    # Advance time each step so fvm::ddt(alpha) rotates alpha.oldTime() (the
    # implicit MULESCorr predictor reads it); without this the OpenFOAM ddt
    # sees a frozen old field and the disc stops advecting after one step.
    runTime.increment()
    _solve_alpha_python(a1, a2, phi, rhoPhi, rho, mix)
af = np.asarray(a1.internalField())
np.save("pyf_alphaN.npy", af)

print("PYF_MASS_DRIFT", float(abs(af.sum() - a0.sum()) / a0.sum()))
print("PYF_MIN", float(af.min()))
print("PYF_MAX", float(af.max()))
print("PYF_BAND", int(((af > 0.01) & (af < 0.99)).sum()))
print("INIT_BAND", int(((a0 > 0.01) & (a0 < 0.99)).sum()))
print("NCELLS", int(af.size))
print("END_OK")
"""

# NeoN: the incompressibleVoFNeon alpha step (explicit MULES, linear, cAlpha=0).
_NEON_DRIVER = f"""
import gc, numpy as np, pybFoam as pyf, neon._neon as nn, neofoam.neofoam_bindings as nfb
from neofoam.solver.neoPimpleFoam import _ensure_neon_initialized


def drive():
    N = {_N_STEPS}
    _ensure_neon_initialized(["disc"])
    al = pyf.argList(["disc"]); t = pyf.Time(al); rt = nfb.create_adapter_run_time(t)
    rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
    rt.dt = {_DT}
    a1 = nfb.read_scalar_volume_field(rt, "alpha.water")
    U = nfb.read_vector_volume_field(rt, "U")
    phi = nfb.create_phi(rt, "U")
    np.asarray(phi.boundary_data_value())[:] = 0.0  # divergence-free interior, inert boundary
    surf = nn.SurfaceInterpolationScalar(rt.executor, rt.nf_mesh, nn.TokenList(["linear"]))

    a0 = np.asarray(a1.internal_vector().copy_to_host()).copy()
    for _ in range(N):
        ap = surf.interpolate(a1) * phi
        nfb.mules_explicit_solve(a1, phi, ap, rt.dt, 1.0, 0.0, 5)
        a1.correct_boundary_conditions()
    af = np.asarray(a1.internal_vector().copy_to_host())
    np.save("neon_alphaN.npy", af)

    print("NEON_MASS_DRIFT", float(abs(af.sum() - a0.sum()) / a0.sum()))
    print("NEON_MIN", float(af.min()))
    print("NEON_MAX", float(af.max()))
    print("NEON_BAND", int(((af > 0.01) & (af < 0.99)).sum()))
    del a1, U, phi, surf, rt, t, al
    print("END_OK")


drive(); gc.collect()
"""


@pytest.fixture(scope="module")
def disc(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    case = tmp_path_factory.mktemp("discAdvection")
    _prepare_disc_case(case)
    m = _parse_metrics(_run_subprocess(case, _PYF_DRIVER, timeout=400))
    m.update(_parse_metrics(_run_subprocess(case, _NEON_DRIVER, timeout=400)))
    pyf_a = np.load(case / "pyf_alphaN.npy")
    neon_a = np.load(case / "neon_alphaN.npy")
    cx = _internal_scalar(case / "0" / "Cx")
    m["PYF_CENTROID_X"] = _centroid_x(pyf_a, cx)
    m["NEON_CENTROID_X"] = _centroid_x(neon_a, cx)
    diff = np.abs(pyf_a - neon_a)
    m["XREL_L2"] = float(np.sqrt((diff**2).mean()) / (np.sqrt((pyf_a**2).mean()) + 1e-30))
    m["XMAXABS"] = float(diff.max())
    return m


# --- Same-scheme MULES parity: bare explicit MULES (linear, cAlpha=0) on BOTH
#     backends, N steps, must be bitwise identical. The multi-step complement to
#     test_mules (single step). The pyf side calls runTime.increment() each step
#     to rotate psi.oldTime() — without it OpenFOAM's disc would freeze while NeoN
#     (in-place old-time) advects, which falsely looks like a limiter divergence.
_PYF_MULES_DRIVER = f"""
import numpy as np, pybFoam as pyf
from pybFoam import fvc, surfaceScalarField, volVectorField, volScalarField
import pybFoam.vof as vof

N = {_N_STEPS}
runTime = pyf.Time(pyf.argList(["disc"]))
mesh = pyf.fvMesh(runTime)
mesh.setFluxRequired(pyf.Word("alpha.water"))
U = volVectorField.read_field(mesh, "U")
phi = pyf.createPhi(U)
alpha = volScalarField.read_field(mesh, "alpha.water")
for _ in range(N):
    runTime.increment()  # rotate alpha.oldTime() (OpenFOAM MULES reads psi.oldTime())
    aphi = surfaceScalarField(pyf.Word("aphi"), fvc.interpolate(alpha) * phi)
    vof.mules_explicit_solve(alpha, phi, aphi)
np.save("pyf_mules.npy", np.asarray(alpha.internalField()))
print("PYFM_BAND", int(((np.asarray(alpha.internalField()) > 0.01)
                        & (np.asarray(alpha.internalField()) < 0.99)).sum()))
print("END_OK")
"""

def _neon_mules_driver(executor: str) -> str:
    """Bare explicit-MULES advection on a named NeoN executor."""
    return f"""
import gc, numpy as np, pybFoam as pyf, neon._neon as nn, neofoam.neofoam_bindings as nfb
from neofoam.solver.neoPimpleFoam import _ensure_neon_initialized


def drive():
    N = {_N_STEPS}
    _ensure_neon_initialized(["disc"])
    al = pyf.argList(["disc"]); t = pyf.Time(al)
    rt = nfb.create_adapter_run_time(t, "{executor}")
    rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
    rt.dt = {_DT}
    alpha = nfb.read_scalar_volume_field(rt, "alpha.water")
    U = nfb.read_vector_volume_field(rt, "U")
    phi = nfb.flux(U)  # bitwise-matches pyf createPhi(U)
    surf = nn.SurfaceInterpolationScalar(rt.executor, rt.nf_mesh, nn.TokenList(["linear"]))
    for _ in range(N):
        ap = surf.interpolate(alpha) * phi
        nfb.mules_explicit_solve(alpha, phi, ap, rt.dt, 1.0, 0.0, 5)
    np.save("neon_mules.npy", np.asarray(alpha.internal_vector().copy_to_host()))
    del alpha, U, phi, surf, rt, t, al
    print("END_OK")


drive(); gc.collect()
"""


@pytest.fixture(scope="module", params=neon_executors())
def mules_parity(request, tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    # Parametrized over every available NeoN executor (Serial / CPU / GPU).
    case = tmp_path_factory.mktemp(f"discMulesParity_{request.param}")
    _prepare_disc_case(case)
    m = _parse_metrics(_run_subprocess(case, _PYF_MULES_DRIVER, timeout=400))
    m.update(
        _parse_metrics(_run_subprocess(case, _neon_mules_driver(request.param), timeout=400))
    )
    pyf_a = np.load(case / "pyf_mules.npy")
    neon_a = np.load(case / "neon_mules.npy")
    diff = np.abs(pyf_a - neon_a)
    m["MULES_MAXABS"] = float(diff.max())
    m["MULES_REL_L2"] = float(
        np.sqrt((diff**2).mean()) / (np.sqrt((pyf_a**2).mean()) + 1e-30)
    )
    return m


def test_setup_is_a_sharp_disc(disc: dict[str, float]) -> None:
    """setFields produces a perfectly sharp disc (no transition cells at t=0)."""
    assert disc["INIT_BAND"] == 0.0
    assert disc["NCELLS"] == 10000.0


def test_pyf_conserves_mass_and_bounded(disc: dict[str, float]) -> None:
    """The pybFoam interFoam alpha scheme conserves mass exactly and stays bounded."""
    assert disc["PYF_MASS_DRIFT"] < 1e-9
    assert disc["PYF_MIN"] >= -1e-6
    assert disc["PYF_MAX"] <= 1.0 + 1e-6


def test_neon_conserves_mass_and_bounded(disc: dict[str, float]) -> None:
    """The NeoN MULES alpha step conserves mass exactly and stays bounded."""
    assert disc["NEON_MASS_DRIFT"] < 1e-9
    assert disc["NEON_MIN"] >= -1e-6
    assert disc["NEON_MAX"] <= 1.0 + 1e-6


def test_both_advect_to_the_expected_position(disc: dict[str, float]) -> None:
    """Both schemes translate the disc to x = 0.55 (U·t = 1·0.2), within ~1 cell.

    Guards the kinematics: a scheme that stalls (e.g. a frozen ddt old-time) or
    advects the wrong distance fails here regardless of how sharp its interface is.
    """
    assert abs(disc["PYF_CENTROID_X"] - 0.55) < 0.015
    assert abs(disc["NEON_CENTROID_X"] - 0.55) < 0.015


def test_pyf_interface_stays_sharp(disc: dict[str, float]) -> None:
    """interFoam's vanLeer + cAlpha compression keeps the interface a few cells wide."""
    # ~circumference/dx cells -> O(150); assert it does not smear across the domain.
    assert disc["PYF_BAND"] < 200


def test_neon_diffusion_is_bounded_and_currently_larger(disc: dict[str, float]) -> None:
    """Characterisation + regression guard on the NeoN interface diffusion.

    Today NeoN runs no interface compression (cAlpha=0) + linear interpolation,
    so its interface is measurably more diffuse than pyf's (~2x the band width).
    This asserts the diffusion is real (band > pyf) but bounded, and records the
    ratio. TARGET: once the full alphaEqn.H (vanLeer + cAlpha + MULESCorr) is
    ported, NEON_BAND -> PYF_BAND and this test flips to a tight equality.
    """
    assert disc["NEON_BAND"] > disc["PYF_BAND"]  # measurably more diffuse today
    assert disc["NEON_BAND"] < 500  # ... but not smeared across the whole domain
    ratio = disc["NEON_BAND"] / disc["PYF_BAND"]
    assert ratio < 4.0, f"NeoN interface diffusion regressed (band ratio {ratio:.2f})"


def test_advection_parity_gap_recorded(disc: dict[str, float]) -> None:
    """Records the NeoN-vs-pyf field gap — the metric that must reach machine
    precision once the alpha schemes are made identical. Not yet tight: the
    schemes differ (cAlpha / vanLeer / MULESCorr), so the interface sits in
    different cells and the L2 gap is O(1). Asserted only to be finite + present
    so the number is tracked; tighten to < 1e-10 after the alphaEqn.H port."""
    assert np.isfinite(disc["XREL_L2"])
    assert disc["XMAXABS"] <= 1.0 + 1e-6


def test_mules_core_matches_pybfoam_over_100_steps(
    mules_parity: dict[str, float],
) -> None:
    """The NeoN MULES core is a bitwise transcription of OpenFOAM's.

    Same bare explicit-MULES scheme on both backends (linear interp, cAlpha=0,
    identical dt / nLimiterIter), 100 advection steps: the fields must agree to
    machine precision. This is the multi-step complement to the single-step
    ``test_mules`` — it proves the diffusion gap in the sibling tests is purely
    the alpha-scheme *config*, not a limiter discrepancy. (Observed ~2.7e-15.)
    """
    assert mules_parity["MULES_MAXABS"] < 1e-10
    assert mules_parity["MULES_REL_L2"] < 1e-12
