# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MULES parity + boundedness + conservation for the NeoN-core FCT limiter.

Value-matches NeoN's ``nfb.mules_explicit_solve`` against pybFoam's
``vof.mules_explicit_solve`` (OpenFOAM ``MULES::explicitSolve``) on identical
inputs on the shared damBreak mesh, via the two-subprocess ``.npy`` harness.

The parity config is load-bearing: a uniform interior velocity U=(1,0,0) gives an
interior face flux phi_f = U.Sf = Sf_x (divergence-free), and all boundary face
fluxes are zero (U is zero on every patch). The zeroed boundary flux makes
OpenFOAM's coupled/fixesValue boundary branches inert (boundary phiCorr == 0, no
extrema widening), so the NeoN simplified path compares like-for-like against the
full-boundary OpenFOAM algorithm — and the telescoping mass identity
Sum((a_new-a_old)*V) == -dt*Sum_boundary(alphaPhi) == 0 holds exactly.
"""

from __future__ import annotations

import os

import pytest

from vof_parity_harness import (
    NEON_VOF_STATE,
    REL_ERR,
    _parse_metrics,
    _run_subprocess,
    prepare_case,
    run_parity,
)

os.environ.setdefault("FOAM_SIGFPE", "false")

# damBreak fvSolution: solvers."alpha.water.*" { nLimiterIter 5; }
_N_LIMITER_ITER = 5


# pyf reference: uniform interior U=(1,0,0) -> divergence-free interior flux, zero
# boundary flux; high-order alphaPhiUn = interpolate(alpha)*phi; then MULES.
_PYF_REF = r"""
import numpy as np
import pybFoam as pyf
from pybFoam import fvc, surfaceScalarField, volScalarField
import pybFoam.vof as vof

runTime = pyf.Time(pyf.argList(["mules"]))
mesh = pyf.fvMesh(runTime)

# Uniform interior velocity (boundary U stays 0 on every patch) -> phi = U.Sf on
# interior faces (== Sf_x), zero on boundary faces.
U = pyf.volVectorField.read_field(mesh, "U")
np.asarray(U.internalField())[:] = [1.0, 0.0, 0.0]
phi = pyf.createPhi(U)

alpha = volScalarField.read_field(mesh, "alpha.water")
np.save("alpha_in.npy", np.asarray(alpha.internalField()).copy())
np.save("phi_int.npy", np.asarray(phi.internalField()).copy())
np.save("V.npy", np.asarray(mesh.V()).copy())
np.save("dt.npy", np.array([mesh.time().deltaTValue()]))

aphi = surfaceScalarField(pyf.Word("alphaPhiUn"), fvc.interpolate(alpha) * phi)
np.save("alphaPhiUn_int.npy", np.asarray(aphi.internalField()).copy())

vof.mules_explicit_solve(alpha, phi, aphi)
np.save("alpha_ref.npy", np.asarray(alpha.internalField()).copy())
np.save("alphaPhi_ref.npy", np.asarray(aphi.internalField()).copy())
print("END_OK")
"""


_NEON = (
    r"""
import gc
import numpy as np
import neon._neon as nn
import neofoam.neofoam_bindings as nfb

N_LIMITER_ITER = """
    + str(_N_LIMITER_ITER)
    + r"""
"""
    + REL_ERR
    + NEON_VOF_STATE
    + r"""
def _drive():
    a0 = np.load("alpha_in.npy")
    phi_int = np.load("phi_int.npy")
    aphiun = np.load("alphaPhiUn_int.npy")
    V = np.load("V.npy")
    dt = float(np.load("dt.npy")[0])
    alpha_ref = np.load("alpha_ref.npy")
    alphaPhi_ref = np.load("alphaPhi_ref.npy")

    s = setup()
    surf = nn.SurfaceInterpolationScalar(
        s.rt.executor, s.rt.nf_mesh, nn.TokenList(["linear"])
    )

    # --- Bit-identical inputs (divergence-free interior, zero boundary flux). ---
    np.asarray(s.alpha1.internal_vector())[:] = a0
    np.asarray(s.phi.internal_vector())[:] = phi_int
    np.asarray(s.phi.boundary_data_value())[:] = 0.0
    alpha_phi = surf.interpolate(s.alpha1) * s.phi
    alpha_phi.name = "alphaPhi"
    np.asarray(alpha_phi.internal_vector())[:] = aphiun
    np.asarray(alpha_phi.boundary_data_value())[:] = 0.0

    nfb.mules_explicit_solve(s.alpha1, s.phi, alpha_phi, dt, 1.0, 0.0, N_LIMITER_ITER)

    a_new = np.asarray(s.alpha1.internal_vector().copy_to_host())
    aphi_new = np.asarray(alpha_phi.internal_vector().copy_to_host())
    print("ALPHA_RELMAX", rel_err(a_new, alpha_ref))
    print("ALPHAPHI_RELMAX", rel_err(aphi_new, alphaPhi_ref))
    print("ALPHA_ABSDIFF", float(np.max(np.abs(a_new - alpha_ref))))
    print("ALPHAPHI_ABSDIFF", float(np.max(np.abs(aphi_new - alphaPhi_ref))))
    print("ALPHA_MIN", float(a_new.min()))
    print("ALPHA_MAX", float(a_new.max()))
    print("MASS_REL", float(abs(np.sum((a_new - a0) * V)) / (np.sum(a0 * V))))

    # --- Degenerate zero-flux case: alpha unchanged, alphaPhi passes through 0. ---
    np.asarray(s.alpha1.internal_vector())[:] = a0
    np.asarray(s.phi.internal_vector())[:] = 0.0
    np.asarray(s.phi.boundary_data_value())[:] = 0.0
    azp = surf.interpolate(s.alpha1) * s.phi
    azp.name = "azp"
    np.asarray(azp.internal_vector())[:] = 0.0
    np.asarray(azp.boundary_data_value())[:] = 0.0
    nfb.mules_explicit_solve(s.alpha1, s.phi, azp, dt, 1.0, 0.0, N_LIMITER_ITER)
    az = np.asarray(s.alpha1.internal_vector().copy_to_host())
    print("ALPHA_ZERO_DRIFT", float(np.max(np.abs(az - a0))))
    print(
        "ALPHAPHI_ZERO_ABSMAX",
        float(np.max(np.abs(np.asarray(azp.internal_vector().copy_to_host())))),
    )

    del surf, alpha_phi, azp, s
    print("END_OK")


_drive()
gc.collect()
"""
)


# NeoN-only unit (no pyf): drives the MULES boundary-face scatter that every parity /
# conservation test leaves value-inert (they zero the boundary alphaPhi). Feeds a uniform
# nonzero boundary alphaPhi on a closed cell with zeroed interior flux, so only the
# boundary efflux is active, and checks the telescoping identity
# Sum((a_new-a_old)*V) == -dt*Sum_boundary(alphaPhi) != 0 (validates owner index + sign).
_MULES_BND_DRIVER = (
    r"""
import gc
import numpy as np
import neon._neon as nn
import neofoam.neofoam_bindings as nfb
"""
    + NEON_VOF_STATE
    + r"""
BND_VAL = 1.0e-3


def host(f):
    return np.asarray(f.internal_vector().copy_to_host())


def _drive():
    s = setup()
    V = np.asarray(s.rt.nf_mesh.cell_volumes.copy_to_host())
    surf = nn.SurfaceInterpolationScalar(
        s.rt.executor, s.rt.nf_mesh, nn.TokenList(["linear"])
    )

    np.asarray(s.alpha1.internal_vector())[:] = 0.5
    s.alpha1.correct_boundary_conditions()
    np.asarray(s.phi.internal_vector())[:] = 0.0
    np.asarray(s.phi.boundary_data_value())[:] = 0.0

    alpha_phi = surf.interpolate(s.alpha1) * s.phi
    alpha_phi.name = "alphaPhi"
    np.asarray(alpha_phi.internal_vector())[:] = 0.0
    apB = np.asarray(alpha_phi.boundary_data_value())
    apB[:] = BND_VAL
    minus_dt_bnd = -float(s.rt.dt) * float(np.sum(apB))

    a_before = host(s.alpha1).copy()
    nfb.mules_explicit_solve(s.alpha1, s.phi, alpha_phi, s.rt.dt, 1.0, 0.0, 5)
    a_after = host(s.alpha1)
    print("DSUM_ALPHAV", float(np.sum((a_after - a_before) * V)))
    print("MINUS_DT_BND", minus_dt_bnd)

    del surf, alpha_phi, s
    print("END_OK")


_drive()
gc.collect()
"""
)


@pytest.fixture(scope="module")
def mules(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    case = tmp_path_factory.mktemp("damBreak_mules")
    prepare_case(case)
    return run_parity(case, _PYF_REF, _NEON)


@pytest.fixture(scope="module")
def mules_bnd(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    case = tmp_path_factory.mktemp("damBreak_mules_bnd")
    prepare_case(case)
    return _parse_metrics(_run_subprocess(case, _MULES_BND_DRIVER))


def test_mules_matches_foam(mules: dict[str, float]) -> None:
    # Identical inputs -> deterministic MULES -> both the limited alphaPhi and the
    # advanced alpha match near machine precision (single tight relative bound, no
    # abs-floor disjunction that would quietly loosen alphaPhi to ~rel 1e-6).
    assert mules["ALPHAPHI_RELMAX"] < 1e-8
    assert mules["ALPHA_RELMAX"] < 1e-8


def test_mules_bounds_alpha(mules: dict[str, float]) -> None:
    # The unlimited linear flux overshoots at the sharp interface; the FCT limiter
    # pulls alpha back into [0,1] WITHOUT any clamp.
    assert mules["ALPHA_MIN"] >= -1e-12
    assert mules["ALPHA_MAX"] <= 1.0 + 1e-12


def test_mules_conserves_mass(mules: dict[str, float]) -> None:
    # Zero boundary alphaPhi -> internal faces telescope -> Sum((a_new-a_old)*V) == 0.
    assert mules["MASS_REL"] < 1e-9


def test_mules_zero_flux_is_passthrough(mules: dict[str, float]) -> None:
    # phi == 0 everywhere: alpha unchanged and alphaPhi stays 0 (guards the
    # ROOTVSMALL divide and the lambda initialisation).
    assert mules["ALPHA_ZERO_DRIFT"] == pytest.approx(0.0, abs=1e-14)
    assert mules["ALPHAPHI_ZERO_ABSMAX"] == pytest.approx(0.0, abs=1e-14)


def test_mules_boundary_flux_efflux(mules_bnd: dict[str, float]) -> None:
    # A uniform nonzero boundary alphaPhi with zeroed interior flux drives only the
    # boundary scatter: the conservative update telescopes to
    # Sum((a_new-a_old)*V) == -dt*Sum_boundary(alphaPhi), and it is genuinely nonzero
    # (exercises the boundary owner index + sign the parity tests leave inert).
    assert abs(mules_bnd["MINUS_DT_BND"]) > 1e-12
    assert abs(mules_bnd["DSUM_ALPHAV"] - mules_bnd["MINUS_DT_BND"]) < 1e-10 * (
        abs(mules_bnd["MINUS_DT_BND"]) + 1e-30
    )
