# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Reproduce SA turbulence.correct() using pybFoam fvm operators.

Compares three implementations:
1. turbulence.correct() — OF's compiled SA model
2. Manual pybFoam fvm assembly — mirrors OF's correct() C++ code
3. NeoN correct() — our Python/NeoN implementation

Uses the turbulence_context session fixture for shared field state.
"""

from typing import Any

import numpy as np
import pytest

import neon._neon as nn
import pybFoam as pyf
from pybFoam import (
    fvc,
    fvm,
    fvScalarMatrix,
    volScalarField,
    volVectorField,
    volTensorField,
    wallDist,
)

from neofoam import neofoam_bindings as nfb
from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    SpalartAllmarasConfig,
)
from neofoam.solver.incompressibleFluidNeon.models.turbulence.spalartAllmaras import (
    correct as sa_correct_neon,
)


def _of(field: Any) -> np.ndarray:
    return np.array(field.internalField())


def _nn(field: Any) -> np.ndarray:
    return np.asarray(field.internal_vector().__array__())


def _report(name: str, a: np.ndarray, b: np.ndarray, la: str = "A", lb: str = "B") -> None:
    abs_diff = np.max(np.abs(a - b))
    denom = np.max(np.abs(a)) + 1e-30
    within_1 = np.sum(np.abs(a - b) / (np.abs(a) + 1e-30) < 0.01)
    print(f"\n  {name}:")
    print(f"    {la}: min={a.min():.6e}  max={a.max():.6e}  mean={a.mean():.6e}")
    print(f"    {lb}: min={b.min():.6e}  max={b.max():.6e}  mean={b.mean():.6e}")
    print(f"    diff: abs={abs_diff:.6e}  rel={abs_diff / denom:.6e}")
    print(f"    within 1%: {within_1}/{len(a)} ({100*within_1/len(a):.1f}%)")


# ---------------------------------------------------------------------------
# Test: Auxiliary fields match between pybFoam (numpy) and NeoN
# ---------------------------------------------------------------------------


def test_pybfoam_sa_auxiliary_fields(turbulence_context: dict[str, Any]) -> None:
    """Compare SA auxiliary fields: pybFoam tensor ops + numpy vs NeoN."""
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]
    mesh = ctx["U_of"].mesh()

    nuTilda_of = ctx["nuTilda_of"]
    U_of = ctx["U_of"]

    # Compute via numpy (for chi, fv1, fv2, Stilda) + pybFoam (for Omega)
    nuTilda_arr = _of(nuTilda_of)
    d_arr = _of(wallDist.New(mesh).y())

    chi_arr = nuTilda_arr / nu
    fv1_arr = chi_arr**3 / (chi_arr**3 + cfg.Cv1**3)
    fv2_arr = 1.0 - chi_arr / (1.0 + chi_arr * fv1_arr)

    # Omega via pybFoam tensor operations
    gradU_of = volTensorField(fvc.grad(U_of))
    skew_gradU = volTensorField(pyf.skew(gradU_of))
    magSqr_skew = volScalarField(pyf.magSqr(skew_gradU))
    Omega_of_arr = np.sqrt(2.0 * _of(magSqr_skew))

    Stilda_arr = np.maximum(
        Omega_of_arr + fv2_arr * nuTilda_arr / (cfg.kappa * d_arr)**2,
        cfg.Cs * Omega_of_arr,
    )

    # NeoN auxiliary fields
    chi_nn = ctx["nuTilda"] / nu
    fv1_nn = chi_nn**3 / (chi_nn**3 + cfg.Cv1**3)
    fv2_nn = 1.0 - chi_nn / (1.0 + chi_nn * fv1_nn)
    grad_U_nn = nn.exp.grad_field(ctx["U"])
    Omega_nn = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U_nn)), 0.0)
    d_safe = nn.field_max(ctx["d"], 1e-10)
    Stilda_nn = nn.field_max(
        Omega_nn + fv2_nn * ctx["nuTilda"] / (cfg.kappa**2 * d_safe**2),
        cfg.Cs * Omega_nn,
    )

    _report("chi", chi_arr, _nn(chi_nn), "OF", "NeoN")
    _report("fv1", fv1_arr, _nn(fv1_nn), "OF", "NeoN")
    _report("fv2", fv2_arr, _nn(fv2_nn), "OF", "NeoN")
    _report("Omega", Omega_of_arr, _nn(Omega_nn), "OF", "NeoN")
    _report("Stilda", Stilda_arr, _nn(Stilda_nn), "OF", "NeoN")

    np.testing.assert_allclose(_nn(Omega_nn), Omega_of_arr, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(_nn(Stilda_nn), Stilda_arr, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: Manual pybFoam SA equation (simplified: no sources) vs NeoN
# ---------------------------------------------------------------------------


def test_pybfoam_transport_only(turbulence_context: dict[str, Any]) -> None:
    """ddt + div - laplacian(DnuTildaEff) via pybFoam fvm vs NeoN.

    Already proven to match in test_sa_pybfoam_assembly.py but
    included here as baseline.
    """
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]

    # OF
    nuTilda_of = ctx["nuTilda_of"]
    phi_of = pyf.createPhi(ctx["U_of"])
    DnuTildaEff_of = nuTilda_of / cfg.sigma  # simplified: nu omitted for pybFoam compat

    eqn_of = fvScalarMatrix(
        fvm.ddt(nuTilda_of) + fvm.div(phi_of, nuTilda_of) - fvm.laplacian(DnuTildaEff_of, nuTilda_of)
    )
    eqn_of.solve()
    of_result = _of(nuTilda_of).copy()

    # NeoN
    nuTilda_nn = ctx["nuTilda"]
    nn.rotate_old_times(nuTilda_nn)
    DnuTildaEff_nn = nuTilda_nn / cfg.sigma
    interp = nn.SurfaceInterpolationScalar(ctx["rt"].executor, ctx["rt"].nf_mesh, nn.TokenList(["linear"]))
    DnuTildaEff_f = interp.interpolate(DnuTildaEff_nn)

    eqn_nn = nfb.PDESolverScalar(
        nn.imp.ddt(nuTilda_nn) + nn.imp.div(ctx["phi"], nuTilda_nn) - nn.imp.laplacian(DnuTildaEff_f, nuTilda_nn),
        nuTilda_nn, ctx["rt"],
    )
    eqn_nn.solve()
    nn_result = _nn(nuTilda_nn)

    _report("transport only", of_result, nn_result, "OF", "NeoN")
    np.testing.assert_allclose(nn_result, of_result, rtol=1e-3, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: pybFoam SA with field-valued Sp (destruction) vs NeoN
# ---------------------------------------------------------------------------


def test_pybfoam_with_destruction(turbulence_context: dict[str, Any]) -> None:
    """Transport + Sp(Cw1*fw*nuTilda/d², nuTilda) via pybFoam fvm vs NeoN.

    Uses Cw1*nuTilda/d² (fw=1 simplification) as field coefficient.
    """
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]
    mesh = ctx["U_of"].mesh()

    # OF
    nuTilda_of = ctx["nuTilda_of"]
    phi_of = pyf.createPhi(ctx["U_of"])
    DnuTildaEff_of = nuTilda_of / cfg.sigma
    d_of = wallDist.New(mesh).y()

    # Sn = Cw1 * nuTilda / d²  (fw=1 simplification)
    eqn_of = fvScalarMatrix(
        fvm.ddt(nuTilda_of)
        + fvm.div(phi_of, nuTilda_of)
        - fvm.laplacian(DnuTildaEff_of, nuTilda_of)
        + fvm.Sp(cfg.Cw1 * nuTilda_of / pyf.sqr(d_of), nuTilda_of)
    )
    eqn_of.solve()
    of_result = _of(nuTilda_of).copy()

    # NeoN
    nuTilda_nn = ctx["nuTilda"]
    nn.rotate_old_times(nuTilda_nn)
    DnuTildaEff_nn = nuTilda_nn / cfg.sigma
    interp = nn.SurfaceInterpolationScalar(ctx["rt"].executor, ctx["rt"].nf_mesh, nn.TokenList(["linear"]))
    DnuTildaEff_f = interp.interpolate(DnuTildaEff_nn)
    Sn_nn = cfg.Cw1 * nuTilda_nn / (ctx["d"] * ctx["d"])

    eqn_nn = nfb.PDESolverScalar(
        nn.imp.ddt(nuTilda_nn)
        + nn.imp.div(ctx["phi"], nuTilda_nn)
        - nn.imp.laplacian(DnuTildaEff_f, nuTilda_nn)
        + nn.imp.source(Sn_nn, nuTilda_nn),
        nuTilda_nn, ctx["rt"],
    )
    eqn_nn.solve()
    nn_result = _nn(nuTilda_nn)

    _report("with destruction (fw=1)", of_result, nn_result, "OF", "NeoN")
    np.testing.assert_allclose(nn_result, of_result, rtol=1e-3, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: pybFoam SA with full fw destruction
# ---------------------------------------------------------------------------


def test_pybfoam_full_destruction(turbulence_context: dict[str, Any]) -> None:
    """Transport + Sp(Cw1*fw*nuTilda/d², nuTilda) with full fw via pybFoam vs NeoN."""
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]
    mesh = ctx["U_of"].mesh()

    nuTilda_of = ctx["nuTilda_of"]
    phi_of = pyf.createPhi(ctx["U_of"])
    DnuTildaEff_of = nuTilda_of / cfg.sigma
    d_of = wallDist.New(mesh).y()

    # Compute fw via pybFoam
    # r = min(nuTilda/(Stilda*kappa²*d²), 10)
    # Need Stilda — compute via pybFoam
    gradU_of = volTensorField(fvc.grad(ctx["U_of"]))
    Omega_of = volScalarField(pyf.sqrt(2.0 * pyf.magSqr(pyf.skew(gradU_of))))

    # Break every operation into single steps to avoid tmp lifetime issues
    chi_of = volScalarField(nuTilda_of / nu)
    chi3_of = volScalarField(pyf.pow3(chi_of))
    chi3_denom = volScalarField(chi3_of + cfg.Cv1**3)  # uses new __add__(scalar)
    fv1_of = volScalarField(chi3_of / chi3_denom)

    # fv2 = 1 - chi/(1 + chi*fv1)
    chi_fv1 = volScalarField(chi_of * fv1_of)
    one_plus = volScalarField(chi_fv1 + 1.0)  # uses __add__(scalar)
    chi_ratio = volScalarField(chi_of / one_plus)
    fv2_of = volScalarField(-chi_ratio + 1.0)  # -field + scalar = 1 - chi_ratio

    kd2_of = volScalarField(pyf.sqr(cfg.kappa * d_of))
    fv2_term = volScalarField(fv2_of * nuTilda_of / kd2_of)
    Stilda_arg1 = volScalarField(Omega_of + fv2_term)
    Stilda_arg2 = volScalarField(cfg.Cs * Omega_of)
    Stilda_of = volScalarField(pyf.max(Stilda_arg1, Stilda_arg2))

    Stilda_safe = volScalarField(pyf.max(Stilda_of, 1e-10))
    r_denom = volScalarField(Stilda_safe * kd2_of)
    r_raw = volScalarField(nuTilda_of / r_denom)
    r_of = volScalarField(pyf.min(r_raw, 10.0))
    r6_of = volScalarField(pyf.pow6(r_of))
    r6_minus_r = volScalarField(r6_of - r_of)
    g_of = volScalarField(r_of + cfg.Cw2 * r6_minus_r)
    g6_of = volScalarField(pyf.pow6(g_of))
    fw_numer = volScalarField(g6_of + cfg.Cw3**6)
    # (1 + Cw3^6) / (g^6 + Cw3^6) — scalar / field via __rtruediv__
    cw3_6 = cfg.Cw3**6
    fw_denom = volScalarField(g6_of + cw3_6)
    fw_frac = volScalarField((1.0 + cw3_6) / fw_denom)  # scalar / volScalarField via __rtruediv__
    fw_pow = volScalarField(pyf.pow(fw_frac, 1.0/6.0))
    fw_of = volScalarField(g_of * fw_pow)

    # Full destruction: Cw1*fw*nuTilda/d²
    d2_of = volScalarField(pyf.sqr(d_of))
    Sn_of = volScalarField(cfg.Cw1 * fw_of * nuTilda_of / d2_of)

    eqn_of = fvScalarMatrix(
        fvm.ddt(nuTilda_of)
        + fvm.div(phi_of, nuTilda_of)
        - fvm.laplacian(DnuTildaEff_of, nuTilda_of)
        + fvm.Sp(cfg.Cw1 * fw_of * nuTilda_of / d2_of, nuTilda_of)  # pass tmp directly
    )
    eqn_of.solve()
    of_result = _of(nuTilda_of).copy()

    # NeoN — full SA correct but Cb1=0, Cb2=0 (no production, no nonConsDiff)
    nuTilda_nn = ctx["nuTilda"]
    nn.rotate_old_times(nuTilda_nn)
    DnuTildaEff_nn = nuTilda_nn / cfg.sigma
    interp = nn.SurfaceInterpolationScalar(ctx["rt"].executor, ctx["rt"].nf_mesh, nn.TokenList(["linear"]))
    DnuTildaEff_f = interp.interpolate(DnuTildaEff_nn)

    # NeoN fw computation (same as in correct())
    SMALL = 1e-10
    d_safe = nn.field_max(ctx["d"], SMALL)
    chi_nn = nuTilda_nn / nu
    fv1_nn = chi_nn**3 / (chi_nn**3 + cfg.Cv1**3)
    fv2_nn = 1.0 - chi_nn / (1.0 + chi_nn * fv1_nn)
    grad_U_nn = nn.exp.grad_field(ctx["U"])
    Omega_nn = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U_nn)), 0.0)
    Stilda_nn = nn.field_max(
        Omega_nn + fv2_nn * nuTilda_nn / (cfg.kappa**2 * d_safe**2), cfg.Cs * Omega_nn,
    )
    Stilda_safe = nn.field_max(Stilda_nn, SMALL)
    r_nn = nn.field_min(nuTilda_nn / (Stilda_safe * cfg.kappa**2 * d_safe**2), 10.0)
    g_nn = nn.field_max(r_nn + cfg.Cw2 * (r_nn**6 - r_nn), SMALL)
    fw_arg = (1.0 + cfg.Cw3**6) * (g_nn**6 + cfg.Cw3**6) ** (-1.0)
    fw_nn = g_nn * nn.field_pow(nn.field_max(fw_arg, SMALL), 1.0 / 6.0)

    Sn_nn = cfg.Cw1 * fw_nn * nuTilda_nn / d_safe**2

    # Compare Sn coefficients before solve
    _report("Sn_coeff (fw full)", _of(volScalarField(Sn_of)), _nn(Sn_nn), "OF", "NeoN")

    eqn_nn = nfb.PDESolverScalar(
        nn.imp.ddt(nuTilda_nn)
        + nn.imp.div(ctx["phi"], nuTilda_nn)
        - nn.imp.laplacian(DnuTildaEff_f, nuTilda_nn)
        + nn.imp.source(Sn_nn, nuTilda_nn),
        nuTilda_nn, ctx["rt"],
    )
    eqn_nn.solve()
    nn_result = _nn(nuTilda_nn)

    _report("with destruction (full fw)", of_result, nn_result, "OF", "NeoN")
    np.testing.assert_allclose(nn_result, of_result, rtol=1e-3, atol=1e-10)
