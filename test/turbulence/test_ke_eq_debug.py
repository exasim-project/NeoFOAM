# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Debug kEpsilon: compare each coefficient and field between pybFoam and NeoN.

Prints a table of all inputs before solving any equation.
"""

from typing import Any

import numpy as np
import pytest

import neon._neon as nn
import pybFoam as pyf
from pybFoam import fvc, fvm, fvScalarMatrix, volScalarField, volTensorField

from neofoam import neofoam_bindings as nfb
from test_ke_pybfoam_vs_correct import ke_case  # noqa: F401

C1 = 1.44; C2 = 1.92; Cmu = 0.09; SIGMA_EPS = 1.3; SIGMA_K = 1.0; SMALL = 1e-10


def _of(f: Any) -> np.ndarray:
    return np.array(f.internalField())


def _nn(f: Any) -> np.ndarray:
    return np.asarray(f.internal_vector().__array__())


def _compare(name: str, of_arr: np.ndarray, nn_arr: np.ndarray) -> None:
    d = np.max(np.abs(nn_arr - of_arr))
    r = d / (np.max(np.abs(of_arr)) + 1e-30)
    w = np.sum(np.abs(nn_arr - of_arr) / (np.abs(of_arr) + 1e-30) < 0.01)
    print(f"  {name:30s}: abs={d:.2e}  rel={r:.2e}  within1%={w}/{len(of_arr)} ({100*w/len(of_arr):.1f}%)")
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-2, atol=1e-10,
                               err_msg=f"{name}: pybFoam fvm vs NeoN")


def _row(name: str, of_arr: np.ndarray, nn_arr: np.ndarray) -> None:
    d = np.max(np.abs(nn_arr - of_arr))
    r = d / (np.max(np.abs(of_arr)) + 1e-30)
    match = "✓" if r < 1e-10 else f"{r:.2e}"
    print(f"  {name:30s}  OF=[{of_arr.min():.4e},{of_arr.max():.4e}]  "
          f"NeoN=[{nn_arr.min():.4e},{nn_arr.max():.4e}]  rel={match}")


def test_coefficient_table(ke_case: dict[str, Any]) -> None:
    """Print table comparing every coefficient/field between pybFoam and NeoN."""
    turb = ke_case["turb"]
    rt = ke_case["rt_nn"]
    nu = ke_case["nu_value"]

    # OF fields
    k_of = ke_case["k_of"]
    eps_of = ke_case["eps_of"]
    nut_of = ke_case["nut_of"]
    nu_of = volScalarField(turb.nu())
    phi_of = turb.phi()

    # NeoN fields
    k_nn = ke_case["k_nn"]
    eps_nn = ke_case["epsilon_nn"]
    nut_nn = ke_case["nut_nn"]
    phi_nn = ke_case["phi_nn"]

    print("\n=== Input fields ===")
    _row("k", _of(k_of), _nn(k_nn))
    _row("epsilon", _of(eps_of), _nn(eps_nn))
    _row("nut", _of(nut_of), _nn(nut_nn))
    _row("nu", _of(nu_of), np.full(len(_of(nu_of)), nu))

    # phi (internal faces) — materialize tmp
    phi_of_mat = pyf.surfaceScalarField(phi_of)
    phi_of_arr = _of(phi_of_mat)
    phi_nn_arr = _nn(phi_nn)[:len(phi_of_arr)]
    _row("phi (internal)", phi_of_arr, phi_nn_arr)

    print("\n=== Derived coefficients ===")

    # DepsEff
    DepsEff_of_arr = _of(volScalarField(nut_of / SIGMA_EPS + nu_of))
    DepsEff_nn_arr = _nn(nut_nn / SIGMA_EPS + nu)
    _row("DepsEff = nut/sigEps + nu", DepsEff_of_arr, DepsEff_nn_arr)

    # DkEff
    DkEff_of_arr = _of(volScalarField(nut_of / SIGMA_K + nu_of))
    DkEff_nn_arr = _nn(nut_nn / SIGMA_K + nu)
    _row("DkEff = nut/sigK + nu", DkEff_of_arr, DkEff_nn_arr)

    # GbyNu
    gradU_of = volTensorField(fvc.grad(turb.U()))
    GbyNu_of_arr = _of(volScalarField(pyf.doubleInner(gradU_of, pyf.devTwoSymm(gradU_of))))
    gradU_nn = nn.exp.grad_field(ke_case["U_nn"])
    GbyNu_nn_arr = _nn(nn.doubleInner(gradU_nn, nn.devTwoSymm(gradU_nn)))
    _row("GbyNu", GbyNu_of_arr, GbyNu_nn_arr)

    # G = nut * GbyNu
    G_of_arr = _of(nut_of) * GbyNu_of_arr
    G_nn_arr = _nn(nut_nn) * GbyNu_nn_arr
    _row("G = nut * GbyNu", G_of_arr, G_nn_arr)

    # divU
    divU_of_arr = _of(volScalarField(fvc.div(phi_of)))
    divU_nn_arr = _nn(nn.exp.div_flux(phi_nn))
    _row("divU", divU_of_arr, divU_nn_arr)

    print("\n=== Epsilon equation coefficients ===")

    # eps production: C1*Cmu*GbyNu*k
    ep_of = C1 * Cmu * GbyNu_of_arr * _of(k_of)
    ep_nn = C1 * Cmu * GbyNu_nn_arr * _nn(k_nn)
    _row("eps_prod = C1*Cmu*GbyNu*k", ep_of, ep_nn)

    # eps destruction: C2*eps/k
    k_of_safe = np.maximum(_of(k_of), SMALL)
    k_nn_safe = _nn(nn.field_max(k_nn, SMALL))
    ed_of = C2 * _of(eps_of) / k_of_safe
    ed_nn = C2 * _nn(eps_nn) / k_nn_safe
    _row("eps_dest = C2*eps/k", ed_of, ed_nn)

    # eps SuSp coeff: (2/3)*C1*divU
    es_of = (2.0/3.0) * C1 * divU_of_arr
    es_nn = (2.0/3.0) * C1 * divU_nn_arr
    _row("eps_susp = (2/3)*C1*divU", es_of, es_nn)

    print("\n=== k equation coefficients ===")

    # k production: G (explicit)
    _row("k_prod = G", G_of_arr, G_nn_arr)

    # k destruction: eps/k
    kd_of = _of(eps_of) / k_of_safe
    kd_nn = _nn(eps_nn) / k_nn_safe
    _row("k_dest = eps/k", kd_of, kd_nn)

    # k SuSp coeff: (2/3)*divU
    ks_of = (2.0/3.0) * divU_of_arr
    ks_nn = (2.0/3.0) * divU_nn_arr
    _row("k_susp = (2/3)*divU", ks_of, ks_nn)

    print("\n=== Differences (should all be zero) ===")
    checks = [
        ("k", _of(k_of), _nn(k_nn)),
        ("epsilon", _of(eps_of), _nn(eps_nn)),
        ("nut", _of(nut_of), _nn(nut_nn)),
        ("phi", phi_of_arr, phi_nn_arr),
        ("DepsEff", DepsEff_of_arr, DepsEff_nn_arr),
        ("DkEff", DkEff_of_arr, DkEff_nn_arr),
        ("GbyNu", GbyNu_of_arr, GbyNu_nn_arr),
        ("G", G_of_arr, G_nn_arr),
        ("divU", divU_of_arr, divU_nn_arr),
        ("eps_prod", ep_of, ep_nn),
        ("eps_dest", ed_of, ed_nn),
        ("eps_susp", es_of, es_nn),
        ("k_prod", G_of_arr, G_nn_arr),
        ("k_dest", kd_of, kd_nn),
        ("k_susp", ks_of, ks_nn),
    ]
    all_ok = True
    for name, a, b in checks:
        d = np.max(np.abs(a - b))
        print(f"  {name:20s}: max|diff| = {d:.2e}")
        if d / (np.max(np.abs(a)) + 1e-30) > 1e-10:
            all_ok = False
            print(f"    ^^^ MISMATCH rel={d / (np.max(np.abs(a)) + 1e-30):.2e}")

    assert all_ok, "Some coefficients differ"

    # Compare face-interpolated epsilon (the solve field itself)
    print("\n=== Face-interpolated fields ===")
    interp = nn.SurfaceInterpolationScalar(rt.executor, rt.nf_mesh, nn.TokenList(["linear"]))

    # Correct BCs first
    eps_nn.correct_boundary_conditions()

    eps_nn_surf = interp.interpolate(eps_nn)
    eps_of_surf = pyf.fvc.interpolate(eps_of)
    # OF returns tmp_surfaceScalarField — need to materialize
    eps_of_surf_mat = pyf.surfaceScalarField(eps_of_surf)
    of_surf_arr = _of(eps_of_surf_mat)
    nn_surf_arr = _nn(eps_nn_surf)[:len(of_surf_arr)]
    _row("epsilon_f (internal faces)", of_surf_arr, nn_surf_arr)

    d = np.max(np.abs(nn_surf_arr - of_surf_arr))
    r = d / (np.max(np.abs(of_surf_arr)) + 1e-30)
    print(f"\n  epsilon_f diff: abs={d:.2e}  rel={r:.2e}")

    # Also compare full surface field (including boundary faces)
    nn_full = _nn(eps_nn_surf)
    print(f"  NeoN surf total size: {len(nn_full)} (internal: {len(of_surf_arr)}, boundary: {len(nn_full)-len(of_surf_arr)})")
    if len(nn_full) > len(of_surf_arr):
        nn_bnd = nn_full[len(of_surf_arr):]
        print(f"  NeoN boundary faces: min={nn_bnd.min():.4e} max={nn_bnd.max():.4e}")

    # DepsEff surface field boundary check
    print("\n=== DepsEff surface field boundary faces ===")
    DepsEff_nn_field = nut_nn / SIGMA_EPS + nu
    DepsEff_nn_field.correct_boundary_conditions()
    DepsEff_nn_surf = interp.interpolate(DepsEff_nn_field)
    deps_full = _nn(DepsEff_nn_surf)
    deps_internal = deps_full[:len(of_surf_arr)]
    deps_boundary = deps_full[len(of_surf_arr):]
    print(f"  DepsEff_f internal: [{deps_internal.min():.6e}, {deps_internal.max():.6e}]")
    print(f"  DepsEff_f boundary: [{deps_boundary.min():.6e}, {deps_boundary.max():.6e}]")

    # Compare with DepsEff cell values at boundary (should match for zeroGradient)
    # The boundary face coefficient should equal the adjacent cell coefficient
    # DepsEff internal: [2.5e-5, 7.8e-4]
    # If boundary DepsEff is 0 or garbage, the laplacian BC contribution is wrong
    print(f"  DepsEff internal field: [{DepsEff_nn_arr.min():.6e}, {DepsEff_nn_arr.max():.6e}]")

    # Check ratio: boundary should be in same range as internal for zeroGradient
    if deps_boundary.min() < DepsEff_nn_arr.min() * 0.1:
        print(f"  WARNING: boundary DepsEff min ({deps_boundary.min():.6e}) much smaller than internal min ({DepsEff_nn_arr.min():.6e})")

    print()


def test_epsilon_boundary_data(ke_case: dict[str, Any]) -> None:
    """Check epsilon boundary face values before and after correctBC."""
    eps_nn = ke_case["epsilon_nn"]
    nn_int = _nn(eps_nn)

    bd_before = np.asarray(eps_nn.boundary_data().value().__array__()).copy()
    eps_nn.correct_boundary_conditions()
    bd_after = np.asarray(eps_nn.boundary_data().value().__array__()).copy()

    print(f"\n  epsilon internal: min={nn_int.min():.4e} max={nn_int.max():.4e}")
    print(f"  boundary BEFORE correctBC: min={bd_before.min():.4e} max={bd_before.max():.4e}")
    print(f"  boundary AFTER correctBC:  min={bd_after.min():.4e} max={bd_after.max():.4e}")
    print(f"  boundary changed: {not np.allclose(bd_before, bd_after)}")

    # For zeroGradient, boundary values should be in internal range
    assert bd_after.min() > 0.5, (
        f"Boundary epsilon min={bd_after.min():.4e} too small for zeroGradient "
        f"(internal range [{nn_int.min():.4e}, {nn_int.max():.4e}])"
    )


def test_epsilon_term_by_term(ke_case: dict[str, Any]) -> None:
    """Add one term at a time, compare pybFoam fvm vs NeoN after each."""
    turb = ke_case["turb"]
    eps_of = ke_case["eps_of"]
    k_of = ke_case["k_of"]
    nut_of = ke_case["nut_of"]
    nu_of = volScalarField(turb.nu())
    phi_of = turb.phi()

    rt = ke_case["rt_nn"]
    eps_nn = ke_case["epsilon_nn"]
    k_nn = ke_case["k_nn"]
    nut_nn = ke_case["nut_nn"]
    phi_nn = ke_case["phi_nn"]
    nu = ke_case["nu_value"]

    DepsEff_of = nut_of / SIGMA_EPS + nu_of
    DepsEff_nn = nut_nn / SIGMA_EPS + nu
    interp = nn.SurfaceInterpolationScalar(rt.executor, rt.nf_mesh, nn.TokenList(["linear"]))
    DepsEff_f = interp.interpolate(DepsEff_nn)
    k_safe = nn.field_max(k_nn, SMALL)
    ones = nn.ScalarVolumeField(rt.executor, "o", rt.nf_mesh)
    nn.fill(ones.internal_vector(), 1.0)
    zero_field = nn.ScalarVolumeField(rt.executor, "z", rt.nf_mesh)
    nn.fill(zero_field.internal_vector(), 0.0)

    gradU_of = volTensorField(fvc.grad(turb.U()))
    GbyNu_of = volScalarField(pyf.doubleInner(gradU_of, pyf.devTwoSymm(gradU_of)))
    divU_of = volScalarField(fvc.div(pyf.surfaceScalarField(turb.phi())))
    gradU_nn = nn.exp.grad_field(ke_case["U_nn"])
    GbyNu_nn = nn.doubleInner(gradU_nn, nn.devTwoSymm(gradU_nn))
    divU_nn = nn.exp.div_flux(phi_nn)

    nu_ds = pyf.dimensionedScalar("nu", pyf.dimViscosity, nu)
    nu_surf = nfb.create_uniform_surface_field(rt, "nu_s", nu)

    def solve_of(build_eqn):
        saved = volScalarField(eps_of)
        build_eqn().solve()
        result = _of(eps_of).copy()
        eps_of.assign(saved)
        return result

    def solve_nn(build_eqn):
        nn.rotate_old_times(eps_nn)
        # Correct BCs AFTER rotate (rotate may reset boundary data)
        eps_nn.correct_boundary_conditions()
        k_nn.correct_boundary_conditions()
        nut_nn.correct_boundary_conditions()
        build_eqn().solve()
        return _nn(eps_nn).copy()

    print()

    # 1. ddt
    of1 = solve_of(lambda: fvScalarMatrix(fvm.ddt(eps_of)))
    nn1 = solve_nn(lambda: nfb.PDESolverScalar(
        nn.imp.ddt(eps_nn) + nn.imp.source(zero_field, eps_nn), eps_nn, rt))
    _compare("1. ddt", of1, nn1)

    # 2. ddt + div
    of2 = solve_of(lambda: fvScalarMatrix(fvm.ddt(eps_of) + fvm.div(phi_of, eps_of)))
    nn2 = solve_nn(lambda: nfb.PDESolverScalar(
        nn.imp.ddt(eps_nn) + nn.imp.div(phi_nn, eps_nn), eps_nn, rt))
    _compare("2. ddt + div", of2, nn2)

    # 3a. ddt + div - lap(const nu)
    of3a = solve_of(lambda: fvScalarMatrix(
        fvm.ddt(eps_of) + fvm.div(phi_of, eps_of) - fvm.laplacian(nu_ds, eps_of)))
    nn3a = solve_nn(lambda: nfb.PDESolverScalar(
        nn.imp.ddt(eps_nn) + nn.imp.div(phi_nn, eps_nn) - nn.imp.laplacian(nu_surf, eps_nn),
        eps_nn, rt))
    _compare("3a. lap(const nu)", of3a, nn3a)

    # 3b. ddt + div - lap(nut/sigma)
    DepsEff_no_nu_f = interp.interpolate(nut_nn / SIGMA_EPS)
    of3b = solve_of(lambda: fvScalarMatrix(
        fvm.ddt(eps_of) + fvm.div(phi_of, eps_of) - fvm.laplacian(nut_of / SIGMA_EPS, eps_of)))
    nn3b = solve_nn(lambda: nfb.PDESolverScalar(
        nn.imp.ddt(eps_nn) + nn.imp.div(phi_nn, eps_nn) - nn.imp.laplacian(DepsEff_no_nu_f, eps_nn),
        eps_nn, rt))
    _compare("3b. lap(nut/sig)", of3b, nn3b)

    # 3c. ddt + div - lap(DepsEff)
    of3c = solve_of(lambda: fvScalarMatrix(
        fvm.ddt(eps_of) + fvm.div(phi_of, eps_of) - fvm.laplacian(DepsEff_of, eps_of)))
    nn3c = solve_nn(lambda: nfb.PDESolverScalar(
        nn.imp.ddt(eps_nn) + nn.imp.div(phi_nn, eps_nn) - nn.imp.laplacian(DepsEff_f, eps_nn),
        eps_nn, rt))
    _compare("3c. lap(DepsEff)", of3c, nn3c)

    # 4. + Sp(dest)
    of4 = solve_of(lambda: fvScalarMatrix(
        fvm.ddt(eps_of) + fvm.div(phi_of, eps_of) - fvm.laplacian(DepsEff_of, eps_of)
        + fvm.Sp(C2 * eps_of / k_of, eps_of)))
    nn4 = solve_nn(lambda: nfb.PDESolverScalar(
        nn.imp.ddt(eps_nn) + nn.imp.div(phi_nn, eps_nn) - nn.imp.laplacian(DepsEff_f, eps_nn)
        + nn.imp.source(C2 * eps_nn / k_safe, eps_nn),
        eps_nn, rt))
    _compare("4. + Sp(dest)", of4, nn4)

    # 5. + Su(prod)
    of5 = solve_of(lambda: fvScalarMatrix(
        fvm.ddt(eps_of) + fvm.div(phi_of, eps_of) - fvm.laplacian(DepsEff_of, eps_of)
        + fvm.Sp(C2 * eps_of / k_of, eps_of)
        - fvm.Su((C1 * Cmu) * volScalarField(GbyNu_of * k_of), eps_of)))
    nn5 = solve_nn(lambda: nfb.PDESolverScalar(
        nn.imp.ddt(eps_nn) + nn.imp.div(phi_nn, eps_nn) - nn.imp.laplacian(DepsEff_f, eps_nn)
        + nn.imp.source(C2 * eps_nn / k_safe, eps_nn)
        - nn.exp.source(C1 * GbyNu_nn * Cmu * k_nn, ones),
        eps_nn, rt))
    _compare("5. + Su(prod)", of5, nn5)

    # 6. + SuSp(divU) FULL
    of6 = solve_of(lambda: fvScalarMatrix(
        fvm.ddt(eps_of) + fvm.div(phi_of, eps_of) - fvm.laplacian(DepsEff_of, eps_of)
        + fvm.Sp(C2 * eps_of / k_of, eps_of)
        + fvm.SuSp((2.0/3.0*C1) * divU_of, eps_of)
        - fvm.Su((C1*Cmu) * volScalarField(GbyNu_of * k_of), eps_of)))
    nn6 = solve_nn(lambda: nfb.PDESolverScalar(
        nn.imp.ddt(eps_nn) + nn.imp.div(phi_nn, eps_nn) - nn.imp.laplacian(DepsEff_f, eps_nn)
        + nn.imp.source(C2 * eps_nn / k_safe, eps_nn)
        + nn.imp.source((2.0/3.0*C1) * divU_nn, eps_nn, susp=True)
        - nn.exp.source(C1 * GbyNu_nn * Cmu * k_nn, ones),
        eps_nn, rt))
    _compare("6. FULL", of6, nn6)
