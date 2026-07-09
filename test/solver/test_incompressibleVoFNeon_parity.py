# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""pybFoam value-parity for the VoF *physics* components of incompressibleVoFNeon.

The framework solver ``incompressibleVoFNeon`` composes the same NeoN VoF
primitives the imperative ``neoInterFoam`` does; the face operators (``mag_sf`` /
``sn_grad`` / ``reconstruct``) and the MULES alpha step are parity-tested
elsewhere (``test_vof_parity.py`` / ``test_mules.py``). This module covers the
remaining building blocks the solver's ``alpha_advection`` + ``momentum`` /
``continuity`` operations use, value-matched against their OpenFOAM references on
the shared damBreak mesh through the two-subprocess ``.npy`` harness:

* mixture density   ``rho  = alpha1*rho1 + alpha2*rho2``   (nfb.create_mixture_density)
* mixture viscosity ``mu   = rho * nu_mix``                (nfb.create_mixture_viscosity)
* gravity head      ``gh   = g & C``  /  ``ghf = g & Cf``  (nfb.create_gh / create_ghf)
* surface tension   ``fSigma = sigma*K*snGrad(alpha1)``    (nfb.surface_tension_force)

rho / mu / gh / ghf are algebraic/geometric and match tightly; the surface-tension
force carries a curvature (``K = -div(nHatf)``) whose discretisation is
implementation-sensitive, so it is matched at a looser tolerance (and asserted to
be non-trivially nonzero at the interface, not a zero-vs-zero tautology).
"""

from __future__ import annotations

import os

import pytest

from vof_parity_harness import REL_ERR, prepare_case, run_parity

os.environ.setdefault("FOAM_SIGFPE", "false")


# pyf reference: build the immiscible two-phase mixture on the damBreak mesh and
# write the OpenFOAM reference for each component as a cell/face internalField.
_PYF_REF = r"""
import numpy as np
import pybFoam as pyf
from pybFoam import surfaceScalarField, volScalarField
import pybFoam.multiphase as vof

runTime = pyf.Time(pyf.argList(["parity"]))
mesh = pyf.fvMesh(runTime)

U = pyf.volVectorField.read_field(mesh, "U")
phi = pyf.createPhi(U)
mix = vof.immiscibleIncompressibleTwoPhaseMixture(U, phi)

alpha1 = mix.alpha1()
alpha2 = mix.alpha2()
rho1 = mix.rho1()
rho2 = mix.rho2()

# Mixture density and (dynamic) viscosity. interFoam's mixture has no mu()/rho()
# accessor; rho is built as alpha1*rho1 + alpha2*rho2 (exactly as the pybFoam
# incompressibleVoF solver does) and the dynamic viscosity is rho * nu_mix, where
# mix.nu() is the kinematic mixture viscosity mu/rho.
rho = volScalarField(pyf.Word("rho"), alpha1 * rho1 + alpha2 * rho2)
mu = volScalarField(pyf.Word("mu"), rho * mix.nu())
np.save("ref_rho.npy", np.asarray(rho.internalField()))
np.save("ref_mu.npy", np.asarray(mu.internalField()))

# Gravity head fields gh = (g & C) - ghRef and ghf = (g & Cf) - ghRef (ghRef = 0).
g = pyf.uniformDimensionedVectorField(mesh, "g")
ghRef = pyf.dimensionedScalar("ghRef", g.dimensions() * pyf.dimLength, 0.0)
gh = volScalarField(pyf.Word("gh"), (g & mesh.C()) - ghRef)
ghf = surfaceScalarField(pyf.Word("ghf"), (g & mesh.Cf()) - ghRef)
np.save("ref_gh.npy", np.asarray(gh.internalField()))
np.save("ref_ghf.npy", np.asarray(ghf.internalField()))

# Surface-tension face force sigma*K*snGrad(alpha1). surfaceTensionForce()
# returns a tmp<surfaceScalarField>; materialize it into a concrete field so
# internalField() is available.
stf = surfaceScalarField(pyf.Word("stf"), mix.surfaceTensionForce())
np.save("ref_stf.npy", np.asarray(stf.internalField()))

print("REF_RHO_ABSMAX", float(np.max(np.abs(np.asarray(rho.internalField())))))
print("REF_STF_ABSMAX", float(np.max(np.abs(np.asarray(stf.internalField())))))
print("END_OK")
"""


# NeoN driver: build the same components through the nfb VoF factories the solver
# uses, and rel-err them against the OpenFOAM references. All NeoN handles live in
# _drive so they release before atexit runs nn.finalize() (a lingering Kokkos
# vector at finalize aborts the process).
_NEON = (
    REL_ERR
    + r"""
import gc
import numpy as np
import pybFoam as pyf
import neon._neon as nn
import neofoam.neofoam_bindings as nfb
from neofoam.solver.neoPimpleFoam import _ensure_neon_initialized


def host(f):
    return np.asarray(f.internal_vector().copy_to_host())


def _drive():
    _ensure_neon_initialized(["parity"])
    arg_list = pyf.argList(["parity"])
    run_time = pyf.Time(arg_list)
    rt = nfb.create_adapter_run_time(run_time)
    rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)

    alpha1 = nfb.read_scalar_volume_field(rt, "alpha.water")
    _U = nfb.read_vector_volume_field(rt, "U")
    _phi = nfb.create_phi(rt, "U")
    phase = nfb.read_two_phase_transport_properties(rt)

    rho = nfb.create_mixture_density(rt)
    mu = nfb.create_mixture_viscosity(rt)
    gh = nfb.create_gh(rt)
    ghf = nfb.create_ghf(rt)
    stf = nfb.surface_tension_force(rt, alpha1, phase["sigma"])

    print("RHO_REL", rel_err(host(rho), np.load("ref_rho.npy")))
    print("MU_REL", rel_err(host(mu), np.load("ref_mu.npy")))
    print("GH_REL", rel_err(host(gh), np.load("ref_gh.npy")))
    print("GHF_REL", rel_err(host(ghf), np.load("ref_ghf.npy")))
    print("STF_REL", rel_err(host(stf), np.load("ref_stf.npy")))
    print("STF_NEON_ABSMAX", float(np.max(np.abs(host(stf)))))
    # Sanity ranges: damBreak water/air rho in [1, 1000], gh <= 0 (g points -y).
    print("RHO_MIN", float(host(rho).min()))
    print("RHO_MAX", float(host(rho).max()))

    del alpha1, _U, _phi, rho, mu, gh, ghf, stf, rt, run_time, arg_list
    print("END_OK")


_drive()
gc.collect()
"""
)


@pytest.fixture(scope="module")
def parity(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    case = tmp_path_factory.mktemp("damBreak_vofneon_parity")
    prepare_case(case)
    return run_parity(case, _PYF_REF, _NEON)


def test_mixture_density_matches_pybfoam(parity: dict[str, float]) -> None:
    """rho = alpha1*rho1 + alpha2*rho2 matches OpenFOAM to machine precision."""
    assert parity["RHO_REL"] < 1e-8
    # Physical range on damBreak: air (1) to water (1000).
    assert 0.9 <= parity["RHO_MIN"] <= 1.1
    assert 990.0 <= parity["RHO_MAX"] <= 1001.0


def test_mixture_viscosity_matches_pybfoam(parity: dict[str, float]) -> None:
    """Dynamic mixture viscosity mu = rho*nu_mix matches OpenFOAM."""
    assert parity["MU_REL"] < 1e-8


def test_gravity_head_matches_pybfoam(parity: dict[str, float]) -> None:
    """gh = g&C (cell) and ghf = g&Cf (face) match OpenFOAM geometry."""
    assert parity["GH_REL"] < 1e-8
    assert parity["GHF_REL"] < 1e-8


def test_surface_tension_force_matches_pybfoam(parity: dict[str, float]) -> None:
    """sigma*K*snGrad(alpha1) matches OpenFOAM's mixture.surfaceTensionForce().

    The NeoN curvature (K = -div(nHatf)) reproduces OpenFOAM's to machine
    precision (observed rel-err ~1e-15); the force must also be non-trivially
    nonzero at the interface (not a zero-vs-zero match).
    """
    assert parity["STF_NEON_ABSMAX"] > 1e-3
    assert parity["STF_REL"] < 1e-10
