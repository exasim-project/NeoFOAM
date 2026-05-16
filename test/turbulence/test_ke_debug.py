# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Debug kEpsilon: test each term that differs between pybFoam and NeoN.

Uses the ke_case fixture from test_ke_pybfoam_vs_correct.py.
"""

from typing import Any

import numpy as np
import pytest

import neon._neon as nn
import pybFoam as pyf
from pybFoam import fvc, fvm, fvScalarMatrix, volScalarField, volTensorField

from neofoam import neofoam_bindings as nfb

# Re-use the ke_case fixture
from test_ke_pybfoam_vs_correct import ke_case  # noqa: F401


def _of(f: Any) -> np.ndarray:
    return np.array(f.internalField())


def _nn(f: Any) -> np.ndarray:
    return np.asarray(f.internal_vector().__array__())


def _report(name: str, of_arr: np.ndarray, nn_arr: np.ndarray) -> None:
    abs_diff = np.max(np.abs(nn_arr - of_arr))
    denom = np.max(np.abs(of_arr)) + 1e-30
    w1 = np.sum(np.abs(nn_arr - of_arr) / (np.abs(of_arr) + 1e-30) < 0.01)
    print(f"\n  {name}:")
    print(f"    OF:   min={of_arr.min():.6e}  max={of_arr.max():.6e}")
    print(f"    NeoN: min={nn_arr.min():.6e}  max={nn_arr.max():.6e}")
    print(f"    diff: abs={abs_diff:.6e}  rel={abs_diff / denom:.6e}")
    print(f"    within 1%: {w1}/{len(of_arr)} ({100 * w1 / len(of_arr):.1f}%)")


# ---------------------------------------------------------------------------
# 1. GbyNu = gradU && devTwoSymm(gradU)
# ---------------------------------------------------------------------------


def test_GbyNu(ke_case: dict[str, Any]) -> None:
    """GbyNu matches between pybFoam and NeoN."""
    turb = ke_case["turb"]
    U_of = turb.U()

    gradU_of = volTensorField(fvc.grad(U_of))
    of_arr = _of(volScalarField(pyf.doubleInner(gradU_of, pyf.devTwoSymm(gradU_of))))

    gradU_nn = nn.exp.grad_field(ke_case["U_nn"])
    nn_arr = _nn(nn.doubleInner(gradU_nn, nn.devTwoSymm(gradU_nn)))

    _report("GbyNu", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. G = nut * GbyNu
# ---------------------------------------------------------------------------


def test_G(ke_case: dict[str, Any]) -> None:
    """Production G matches."""
    turb = ke_case["turb"]
    U_of = turb.U()
    nut_of = ke_case["nut_of"]

    gradU_of = volTensorField(fvc.grad(U_of))
    GbyNu_of = volScalarField(pyf.doubleInner(gradU_of, pyf.devTwoSymm(gradU_of)))
    of_arr = _of(volScalarField(nut_of * GbyNu_of))

    gradU_nn = nn.exp.grad_field(ke_case["U_nn"])
    GbyNu_nn = nn.doubleInner(gradU_nn, nn.devTwoSymm(gradU_nn))
    nn_arr = _nn(ke_case["nut_nn"] * GbyNu_nn)

    _report("G", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. divU = fvc::div(phi)
# ---------------------------------------------------------------------------


def test_divU(ke_case: dict[str, Any]) -> None:
    """divU: pybFoam fvc.div(phi) vs NeoN — NeoN currently can't compute this."""
    turb = ke_case["turb"]
    phi_of = turb.phi()

    divU_of = volScalarField(fvc.div(phi_of))
    of_arr = _of(divU_of)

    print(f"\n  divU (pybFoam):")
    print(f"    min={of_arr.min():.6e}  max={of_arr.max():.6e}  mean={of_arr.mean():.6e}")
    print(f"    non-zero cells: {np.sum(np.abs(of_arr) > 1e-10)}/{len(of_arr)}")
    print(f"    L2 norm: {np.sqrt(np.mean(of_arr**2)):.6e}")

    # NeoN: div_flux(phi)
    divU_nn = nn.exp.div_flux(ke_case["phi_nn"])
    nn_arr = _nn(divU_nn)

    _report("divU", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Effective diffusivities
# ---------------------------------------------------------------------------


def test_DkEff(ke_case: dict[str, Any]) -> None:
    """DkEff = nut/sigmaK + nu matches."""
    nu = ke_case["nu_value"]
    sigmaK = 1.0

    of_arr = _of(volScalarField(ke_case["nut_of"] / sigmaK + volScalarField(ke_case["turb"].nu())))
    nn_arr = _nn(ke_case["nut_nn"] / sigmaK + nu)

    _report("DkEff", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-10, atol=1e-15)


# ---------------------------------------------------------------------------
# 5. Epsilon production coefficient: C1*Cmu*GbyNu*k
# ---------------------------------------------------------------------------


def test_eps_production(ke_case: dict[str, Any]) -> None:
    """Epsilon production C1*Cmu*GbyNu*k matches."""
    C1 = 1.44
    Cmu = 0.09
    turb = ke_case["turb"]
    k_of = ke_case["k_of"]

    gradU_of = volTensorField(fvc.grad(turb.U()))
    GbyNu_of = volScalarField(pyf.doubleInner(gradU_of, pyf.devTwoSymm(gradU_of)))
    of_arr = C1 * Cmu * _of(volScalarField(GbyNu_of * k_of))

    gradU_nn = nn.exp.grad_field(ke_case["U_nn"])
    GbyNu_nn = nn.doubleInner(gradU_nn, nn.devTwoSymm(gradU_nn))
    nn_arr = _nn(C1 * GbyNu_nn * Cmu * ke_case["k_nn"])

    _report("eps_production", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. Epsilon destruction coefficient: C2*eps/k
# ---------------------------------------------------------------------------


def test_eps_destruction(ke_case: dict[str, Any]) -> None:
    """Epsilon destruction C2*eps/k matches."""
    C2 = 1.92

    of_arr = C2 * _of(ke_case["eps_of"]) / _of(ke_case["k_of"])

    SMALL = 1e-10
    k_safe = nn.field_max(ke_case["k_nn"], SMALL)
    nn_arr = _nn(C2 * ke_case["epsilon_nn"] / k_safe)

    _report("eps_destruction", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. Epsilon eq WITHOUT SuSp(divU): pybFoam fvm vs NeoN
# ---------------------------------------------------------------------------


def test_epsilon_eq_no_divU(ke_case: dict[str, Any]) -> None:
    """Epsilon equation without SuSp(divU): pybFoam fvm vs NeoN.

    If this matches, the only difference is the missing SuSp term.
    """
    C1 = 1.44
    C2 = 1.92
    Cmu = 0.09
    sigmaEps = 1.3
    SMALL = 1e-10

    turb = ke_case["turb"]
    k_of = ke_case["k_of"]
    eps_of = ke_case["eps_of"]
    nut_of = ke_case["nut_of"]
    nu_of = volScalarField(turb.nu())
    phi_of = turb.phi()

    gradU_of = volTensorField(fvc.grad(turb.U()))
    GbyNu_of = volScalarField(pyf.doubleInner(gradU_of, pyf.devTwoSymm(gradU_of)))
    DepsEff_of = nut_of / sigmaEps + nu_of

    # pybFoam: epsilon eq WITHOUT SuSp
    saved_eps = volScalarField(eps_of)
    eqn_of = fvScalarMatrix(
        fvm.ddt(eps_of)
        + fvm.div(phi_of, eps_of)
        - fvm.laplacian(DepsEff_of, eps_of)
        + fvm.Sp(C2 * eps_of / k_of, eps_of)
        - fvm.Su((C1 * Cmu) * volScalarField(GbyNu_of * k_of), eps_of)
    )
    eqn_of.solve()
    of_arr = _of(eps_of).copy()
    eps_of.assign(saved_eps)  # restore

    # NeoN: same equation (no SuSp)
    nu = ke_case["nu_value"]
    rt = ke_case["rt_nn"]
    epsilon = ke_case["epsilon_nn"]
    k_nn = ke_case["k_nn"]
    nut_nn = ke_case["nut_nn"]
    phi_nn = ke_case["phi_nn"]

    nn.rotate_old_times(epsilon)
    gradU_nn = nn.exp.grad_field(ke_case["U_nn"])
    GbyNu_nn = nn.doubleInner(gradU_nn, nn.devTwoSymm(gradU_nn))

    DepsEff_nn = nut_nn / sigmaEps + nu
    interp = nn.SurfaceInterpolationScalar(rt.executor, rt.nf_mesh, nn.TokenList(["linear"]))
    DepsEff_f = interp.interpolate(DepsEff_nn)

    k_safe = nn.field_max(k_nn, SMALL)
    ones = nn.ScalarVolumeField(rt.executor, "ones", rt.nf_mesh)
    nn.fill(ones.internal_vector(), 1.0)

    eps_prod = C1 * GbyNu_nn * Cmu * k_nn
    eps_dest = C2 * epsilon / k_safe

    epsEqn = nfb.PDESolverScalar(
        nn.imp.ddt(epsilon)
        + nn.imp.div(phi_nn, epsilon)
        - nn.imp.laplacian(DepsEff_f, epsilon)
        + nn.imp.source(eps_dest, epsilon)
        - nn.exp.source(eps_prod, ones),
        epsilon, rt,
    )
    epsEqn.solve()
    nn_arr = _nn(epsilon).copy()

    _report("epsilon (no divU): OF fvm vs NeoN", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-2, atol=1e-10,
                               err_msg="Epsilon eq without SuSp differs between OF and NeoN")


def test_epsilon_eq_with_divU(ke_case: dict[str, Any]) -> None:
    """Epsilon equation WITH SuSp(divU): pybFoam fvm vs NeoN."""
    C1 = 1.44
    C2 = 1.92
    Cmu = 0.09
    sigmaEps = 1.3
    SMALL = 1e-10

    turb = ke_case["turb"]
    k_of = ke_case["k_of"]
    eps_of = ke_case["eps_of"]
    nut_of = ke_case["nut_of"]
    nu_of = volScalarField(turb.nu())
    phi_of = turb.phi()

    gradU_of = volTensorField(fvc.grad(turb.U()))
    GbyNu_of = volScalarField(pyf.doubleInner(gradU_of, pyf.devTwoSymm(gradU_of)))
    DepsEff_of = nut_of / sigmaEps + nu_of
    divU_of = volScalarField(fvc.div(phi_of))

    # pybFoam: epsilon eq WITH SuSp (matches OF)
    saved_eps = volScalarField(eps_of)
    eqn_of = fvScalarMatrix(
        fvm.ddt(eps_of)
        + fvm.div(phi_of, eps_of)
        - fvm.laplacian(DepsEff_of, eps_of)
        + fvm.Sp(C2 * eps_of / k_of, eps_of)
        + fvm.SuSp((2.0 / 3.0 * C1) * divU_of, eps_of)
        - fvm.Su((C1 * Cmu) * volScalarField(GbyNu_of * k_of), eps_of)
    )
    eqn_of.solve()
    of_arr = _of(eps_of).copy()
    eps_of.assign(saved_eps)

    # NeoN: same equation WITH SuSp split
    nu = ke_case["nu_value"]
    rt = ke_case["rt_nn"]
    epsilon = ke_case["epsilon_nn"]
    k_nn = ke_case["k_nn"]
    nut_nn = ke_case["nut_nn"]
    phi_nn = ke_case["phi_nn"]

    nn.rotate_old_times(epsilon)
    gradU_nn = nn.exp.grad_field(ke_case["U_nn"])
    GbyNu_nn = nn.doubleInner(gradU_nn, nn.devTwoSymm(gradU_nn))
    divU_nn = nn.exp.div_flux(phi_nn)

    DepsEff_nn = nut_nn / sigmaEps + nu
    interp = nn.SurfaceInterpolationScalar(rt.executor, rt.nf_mesh, nn.TokenList(["linear"]))
    DepsEff_f = interp.interpolate(DepsEff_nn)

    k_safe = nn.field_max(k_nn, SMALL)
    ones = nn.ScalarVolumeField(rt.executor, "ones2", rt.nf_mesh)
    nn.fill(ones.internal_vector(), 1.0)

    eps_prod = C1 * GbyNu_nn * Cmu * k_nn
    eps_dest = C2 * epsilon / k_safe

    # SuSp split
    eps_susp_coeff = (2.0 / 3.0) * C1 * divU_nn

    epsEqn = nfb.PDESolverScalar(
        nn.imp.ddt(epsilon)
        + nn.imp.div(phi_nn, epsilon)
        - nn.imp.laplacian(DepsEff_f, epsilon)
        + nn.imp.source(eps_dest, epsilon)
        + nn.imp.source(eps_susp_coeff, epsilon, susp=True)
        - nn.exp.source(eps_prod, ones),
        epsilon, rt,
    )
    epsEqn.solve()
    nn_arr = _nn(epsilon).copy()

    _report("epsilon (with divU): OF fvm vs NeoN", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-2, atol=1e-10,
                               err_msg="Epsilon eq with SuSp differs between OF and NeoN")


def test_k_eq_with_divU(ke_case: dict[str, Any]) -> None:
    """k equation WITH SuSp(divU): pybFoam fvm vs NeoN.

    Both solve epsilon first (pybFoam and NeoN independently),
    then compare the k equation using the respective updated epsilon.
    """
    C1 = 1.44
    C2 = 1.92
    Cmu = 0.09
    sigmaK = 1.0
    sigmaEps = 1.3
    SMALL = 1e-10

    turb = ke_case["turb"]
    k_of = ke_case["k_of"]
    eps_of = ke_case["eps_of"]
    nut_of = ke_case["nut_of"]
    nu_of = volScalarField(turb.nu())
    phi_of = turb.phi()

    gradU_of = volTensorField(fvc.grad(turb.U()))
    GbyNu_of = volScalarField(pyf.doubleInner(gradU_of, pyf.devTwoSymm(gradU_of)))
    G_of = volScalarField(nut_of * GbyNu_of)
    divU_of = volScalarField(fvc.div(phi_of))
    DepsEff_of = nut_of / sigmaEps + nu_of
    DkEff_of = nut_of / sigmaK + nu_of

    # --- pybFoam: solve epsilon first, then k ---
    saved_k = volScalarField(k_of)
    saved_eps = volScalarField(eps_of)

    eqn_eps_of = fvScalarMatrix(
        fvm.ddt(eps_of)
        + fvm.div(phi_of, eps_of)
        - fvm.laplacian(DepsEff_of, eps_of)
        + fvm.Sp(C2 * eps_of / k_of, eps_of)
        + fvm.SuSp((2.0 / 3.0 * C1) * divU_of, eps_of)
        - fvm.Su((C1 * Cmu) * volScalarField(GbyNu_of * k_of), eps_of)
    )
    eqn_eps_of.solve()
    pyf.bound(eps_of, pyf.dimensionedScalar("z", pyf.dimViscosity / pyf.dimTime, SMALL))

    # k equation with updated epsilon
    eqn_k_of = fvScalarMatrix(
        fvm.ddt(k_of)
        + fvm.div(phi_of, k_of)
        - fvm.laplacian(DkEff_of, k_of)
        + fvm.Sp(eps_of / k_of, k_of)
        + fvm.SuSp((2.0 / 3.0) * divU_of, k_of)
        - fvm.Su(1.0 * G_of, k_of)
    )
    eqn_k_of.solve()
    pyf.bound(k_of, pyf.dimensionedScalar("z", pyf.dimVelocity * pyf.dimVelocity, SMALL))
    of_k = _of(k_of).copy()
    of_eps = _of(eps_of).copy()

    # Restore OF fields
    k_of.assign(saved_k)
    eps_of.assign(saved_eps)

    # --- NeoN: solve epsilon first, then k ---
    nu = ke_case["nu_value"]
    rt = ke_case["rt_nn"]
    k_nn = ke_case["k_nn"]
    epsilon_nn = ke_case["epsilon_nn"]
    nut_nn = ke_case["nut_nn"]
    phi_nn = ke_case["phi_nn"]

    nn.rotate_old_times(epsilon_nn)
    nn.rotate_old_times(k_nn)

    gradU_nn = nn.exp.grad_field(ke_case["U_nn"])
    GbyNu_nn = nn.doubleInner(gradU_nn, nn.devTwoSymm(gradU_nn))
    G_nn = nut_nn * GbyNu_nn
    divU_nn = nn.exp.div_flux(phi_nn)

    DepsEff_nn = nut_nn / sigmaEps + nu
    DkEff_nn = nut_nn / sigmaK + nu
    interp = nn.SurfaceInterpolationScalar(rt.executor, rt.nf_mesh, nn.TokenList(["linear"]))
    DepsEff_f = interp.interpolate(DepsEff_nn)
    DkEff_f = interp.interpolate(DkEff_nn)

    k_safe = nn.field_max(k_nn, SMALL)
    ones = nn.ScalarVolumeField(rt.executor, "ones3", rt.nf_mesh)
    nn.fill(ones.internal_vector(), 1.0)

    # Solve epsilon
    eps_prod = C1 * GbyNu_nn * Cmu * k_nn
    eps_dest = C2 * epsilon_nn / k_safe
    eps_susp = (2.0 / 3.0) * C1 * divU_nn

    epsEqn = nfb.PDESolverScalar(
        nn.imp.ddt(epsilon_nn)
        + nn.imp.div(phi_nn, epsilon_nn)
        - nn.imp.laplacian(DepsEff_f, epsilon_nn)
        + nn.imp.source(eps_dest, epsilon_nn)
        + nn.imp.source(eps_susp, epsilon_nn, susp=True)
        - nn.exp.source(eps_prod, ones),
        epsilon_nn, rt,
    )
    epsEqn.solve()
    nn.bound(epsilon_nn, SMALL)

    # Compare epsilon before proceeding to k
    nn_eps = _nn(epsilon_nn).copy()
    _report("epsilon (step 1)", of_eps, nn_eps)

    # Solve k with updated epsilon
    k_dest = epsilon_nn / k_safe
    k_susp = (2.0 / 3.0) * divU_nn

    kEqn = nfb.PDESolverScalar(
        nn.imp.ddt(k_nn)
        + nn.imp.div(phi_nn, k_nn)
        - nn.imp.laplacian(DkEff_f, k_nn)
        + nn.imp.source(k_dest, k_nn)
        + nn.imp.source(k_susp, k_nn, susp=True)
        - nn.exp.source(G_nn, ones),
        k_nn, rt,
    )
    kEqn.solve()
    nn.bound(k_nn, SMALL)
    nn_k = _nn(k_nn).copy()

    _report("k (step 2, with updated eps)", of_k, nn_k)

    np.testing.assert_allclose(nn_eps, of_eps, rtol=1e-2, atol=1e-10,
                               err_msg="epsilon with SuSp differs")
    np.testing.assert_allclose(nn_k, of_k, rtol=1e-2, atol=1e-10,
                               err_msg="k with SuSp differs")
