# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""pybFoam ``fvc::`` value-parity clients for the NeoN VoF face operators.

Three clients of ``vof_parity_harness``, each value-matching a NeoN face/cell
operator against its OpenFOAM ``fvc::`` reference on the shared damBreak mesh:

* ``mag_sf``    vs ``mesh.magSf()``          — pure geometry (rel 1e-10, min > 0).
  This face-by-face match also validates that the nfb<->OpenFOAM adapter preserves
  internal-face ordering; if it fails, the ``sn_grad`` client (also a face field,
  same ordering) is expected to fail too — read the two together.
* ``sn_grad``   vs ``fvc::snGrad``           — on a NON-uniform scalar field
  (x-ramp), internal faces (rel 1e-8, |snGrad| > 0). Kills a uniform-input
  tautology: the answer is non-trivially nonzero.
* ``reconstruct`` vs ``fvc::reconstruct``    — cellwise (rel 1e-6).

The ramp is imposed on ``alpha.water`` (zeroGradient walls) rather than ``p_rgh``:
``p_rgh``'s fixedFluxPressure BC makes a bare ``correctBoundaryConditions`` on an
imposed interior field ill-posed, whereas zeroGradient is well-posed and gives a
zero boundary snGrad — so the boundary face-force contribution to ``reconstruct``
vanishes on both sides.
"""

from __future__ import annotations

import os

import pytest

from vof_parity_harness import REL_ERR, prepare_case, run_parity

os.environ.setdefault("FOAM_SIGFPE", "false")


# pyf reference subprocess: writes ref_magSf / ref_snGrad / ref_reconstruct / ref_Cx.
_PYF_REF = r"""
import numpy as np
import pybFoam as pyf
from pybFoam import fvc, surfaceScalarField, volScalarField, volVectorField

runTime = pyf.Time(pyf.argList(["parity"]))
mesh = pyf.fvMesh(runTime)

# Geometry: internal-face area magnitudes.
np.save("ref_magSf.npy", np.asarray(mesh.magSf().internalField()))

# alpha.water's atmosphere inletOutlet BC looks up phi during correction — create
# and register it (from U) so correctBoundaryConditions is well-posed.
U = pyf.volVectorField.read_field(mesh, "U")
phi = pyf.createPhi(U)

# Non-uniform scalar field: x-ramp imposed on alpha.water (zeroGradient walls),
# so correctBoundaryConditions is well-posed and boundary snGrad is ~0.
C = np.asarray(mesh.C().internalField())
np.save("ref_Cx.npy", C[:, 0])
psi = volScalarField.read_field(mesh, "alpha.water")
np.asarray(psi.internalField())[:] = C[:, 0]
psi.correctBoundaryConditions()

sn = surfaceScalarField(pyf.Word("snPsi"), fvc.snGrad(psi))
np.save("ref_snGrad.npy", np.asarray(sn.internalField()))

ff = surfaceScalarField(pyf.Word("ff"), sn * mesh.magSf())
rc = volVectorField(pyf.Word("rc"), fvc.reconstruct(ff))
np.save("ref_reconstruct.npy", np.asarray(rc.internalField()))
print("END_OK")
"""


_NEON = (
    r"""
import gc
import numpy as np
import neon._neon as nn
import neofoam.neofoam_bindings as nfb
from neofoam.solver.neoInterFoam import NeoInterFoam
"""
    + REL_ERR
    + r"""
def _drive():
    ref_magSf = np.load("ref_magSf.npy")
    ref_snGrad = np.load("ref_snGrad.npy")
    ref_recon = np.load("ref_reconstruct.npy")
    Cx = np.load("ref_Cx.npy")

    s = NeoInterFoam(["neoInterFoam"]).setup()

    # mag_sf — pure geometry, face-by-face (also validates internal-face ordering).
    ms = np.asarray(nfb.mag_sf(s.rt).internal_vector().copy_to_host())
    print("MAGSF_RELMAX", rel_err(ms, ref_magSf))
    print("MAGSF_MIN", float(ms.min()))

    # sn_grad on the identical x-ramp scalar field, internal faces.
    np.asarray(s.alpha1.internal_vector())[:] = Cx
    s.alpha1.correct_boundary_conditions()
    sn = np.asarray(nfb.sn_grad(s.alpha1).internal_vector().copy_to_host())
    print("SNGRAD_RELMAX", rel_err(sn, ref_snGrad))
    print("SNGRAD_ABSMAX", float(np.max(np.abs(sn))))

    # reconstruct(sn_grad(psi)*mag_sf); zero the boundary face contribution to
    # match pyf's zeroGradient boundary (snGrad_boundary == 0).
    magsf = nfb.mag_sf(s.rt)
    ff = nfb.sn_grad(s.alpha1) * magsf
    np.asarray(ff.boundary_data_value())[:] = 0.0
    rc = np.asarray(nn.reconstruct(ff).internal_vector().copy_to_host())
    print("RECON_RELMAX", rel_err(rc, ref_recon))
    print("RECON_ABSMAX", float(np.max(np.abs(rc))))
    del magsf, ff, s
    print("END_OK")


_drive()
gc.collect()
"""
)


# --- Surface tension / curvature parity: interface_nhatf + surface_tension_force ---
# Reads the real setFields alpha.water (a sharp interface -> genuine curvature) on BOTH
# sides; no imposed ramp. pyf builds the OpenFOAM mixture and recomputes K/nHatf from the
# on-disk alpha, NeoN computes the same via the new nfb composites; compares internal faces.
_PYF_ST = r"""
import numpy as np
import pybFoam as pyf
from pybFoam import surfaceScalarField, volVectorField
import pybFoam.vof as vof

runTime = pyf.Time(pyf.argList(["stparity"]))
mesh = pyf.fvMesh(runTime)

U = volVectorField.read_field(mesh, "U")
phi = pyf.createPhi(U)
# immiscibleIncompressibleTwoPhaseMixture reads alpha.water + sigma; correct() recomputes
# nHatf/K from the on-disk (setFields) alpha.
mix = vof.immiscibleIncompressibleTwoPhaseMixture(U, phi)
mix.correct()
np.save("nhatf_ref.npy", np.asarray(mix.nHatf().internalField()).copy())
stf = surfaceScalarField(pyf.Word("stf"), mix.surfaceTensionForce())
np.save("stf_ref.npy", np.asarray(stf.internalField()).copy())
print("END_OK")
"""


_NEON_ST = (
    r"""
import gc
import numpy as np
import neofoam.neofoam_bindings as nfb
from neofoam.solver.neoInterFoam import NeoInterFoam
"""
    + REL_ERR
    + r"""
def _drive():
    nhatf_ref = np.load("nhatf_ref.npy")
    stf_ref = np.load("stf_ref.npy")

    s = NeoInterFoam(["neoInterFoam"]).setup()

    nhatf = np.asarray(
        nfb.interface_nhatf(s.rt, s.alpha1).internal_vector().copy_to_host()
    )
    stf = np.asarray(
        nfb.surface_tension_force(s.rt, s.alpha1, s.phase["sigma"])
        .internal_vector()
        .copy_to_host()
    )
    print("NHATF_RELMAX", rel_err(nhatf, nhatf_ref))
    print("NHATF_ABSMAX", float(np.max(np.abs(nhatf))))
    print("STF_RELMAX", rel_err(stf, stf_ref))
    print("STF_ABSMAX", float(np.max(np.abs(stf))))
    del s
    print("END_OK")


_drive()
gc.collect()
"""
)


@pytest.fixture(scope="module")
def parity(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    case = tmp_path_factory.mktemp("damBreak_parity")
    prepare_case(case)
    return run_parity(case, _PYF_REF, _NEON)


@pytest.fixture(scope="module")
def st_parity(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    case = tmp_path_factory.mktemp("damBreak_st_parity")
    prepare_case(case)
    return run_parity(case, _PYF_ST, _NEON_ST)


def test_mag_sf_matches_foam(parity: dict[str, float]) -> None:
    # Face-area magnitudes match mesh.magSf() to machine precision, and are
    # strictly positive (degenerate-face guard / all-zeros guard).
    assert parity["MAGSF_RELMAX"] < 1e-10
    assert parity["MAGSF_MIN"] > 0.0


def test_sn_grad_matches_foam(parity: dict[str, float]) -> None:
    # Internal-face snGrad of a non-uniform (x-ramp) field matches fvc::snGrad,
    # and is non-trivially nonzero (kills the uniform-input tautology).
    assert parity["SNGRAD_RELMAX"] < 1e-8
    assert parity["SNGRAD_ABSMAX"] > 0.0


def test_reconstruct_matches_foam(parity: dict[str, float]) -> None:
    # Cellwise reconstruct(snGrad(psi)*magSf) matches fvc::reconstruct, and the result is
    # non-trivially nonzero. The 1e-6 bar (looser than the sn_grad/mag_sf siblings) is the
    # floor set by NeoN dropping OpenFOAM's empty front/back faces from the 3x3 (regularised
    # in reconstruct.cpp), not drift — RECON_RELMAX is printed so a regression stays visible.
    assert parity["RECON_ABSMAX"] > 0.0
    assert parity["RECON_RELMAX"] < 1e-6


def test_nhatf_matches_foam(st_parity: dict[str, float]) -> None:
    # Interface unit-normal flux nHatf on internal faces matches mixture.nHatf() on the real
    # setFields interface, and is non-trivially nonzero. Gauss-linear grad + linear face
    # interp + orthogonal mesh + identical deltaN -> tight; RELMAX printed to allow tightening.
    assert st_parity["NHATF_ABSMAX"] > 0.0
    assert st_parity["NHATF_RELMAX"] < 1e-8


def test_surface_tension_force_matches_foam(st_parity: dict[str, float]) -> None:
    # Surface-tension face force sigma*K*snGrad(alpha1) on internal faces matches
    # mixture.surfaceTensionForce(), and is non-trivially nonzero.
    assert st_parity["STF_ABSMAX"] > 0.0
    assert st_parity["STF_RELMAX"] < 1e-8
