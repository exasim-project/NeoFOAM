# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Step-by-step comparison: NeoN SA vs pybFoam SA (validated against OF).

Compares each auxiliary field and the final nuTilda result between:
- sa_pybfoam.py (pybFoam fvm, validated to match turbulence.correct())
- spalartAllmaras.py (NeoN imp operators)

Uses the turbulence_context session fixture (provides both backends).
"""

from typing import Any

import numpy as np
import pytest

import neon._neon as nn
import pybFoam as pyf
from pybFoam import fvc, volScalarField, volTensorField, volVectorField, wallDist

from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    SpalartAllmarasConfig,
)

# pybFoam SA functions (validated against turbulence.correct())
from sa_pybfoam import (
    SAConstants,
    compute_chi as of_chi,
    compute_fv1 as of_fv1,
    compute_fv2 as of_fv2,
    compute_Omega as of_Omega,
    compute_Stilda as of_Stilda,
    compute_fw as of_fw,
    compute_DnuTildaEff as of_DnuTildaEff,
)


def _of(field: Any) -> np.ndarray:
    return np.array(field.internalField())


def _nn(field: Any) -> np.ndarray:
    return np.asarray(field.internal_vector().__array__())


def _report(name: str, of_arr: np.ndarray, nn_arr: np.ndarray) -> None:
    abs_diff = np.max(np.abs(nn_arr - of_arr))
    denom = np.max(np.abs(of_arr)) + 1e-30
    w1 = np.sum(np.abs(nn_arr - of_arr) / (np.abs(of_arr) + 1e-30) < 0.01)
    print(f"\n  {name}:")
    print(f"    OF:   min={of_arr.min():.6e}  max={of_arr.max():.6e}")
    print(f"    NeoN: min={nn_arr.min():.6e}  max={nn_arr.max():.6e}")
    print(f"    diff: abs={abs_diff:.6e}  rel={abs_diff / denom:.6e}")
    print(f"    within 1%: {w1}/{len(of_arr)} ({100 * w1 / len(of_arr):.1f}%)")


CFG_OF = SAConstants()
CFG_NN = SpalartAllmarasConfig()
SMALL = 1e-10


# ---------------------------------------------------------------------------
# Step 1: chi
# ---------------------------------------------------------------------------


def test_chi(turbulence_context: dict[str, Any]) -> None:
    ctx = turbulence_context
    nu = ctx["nu_value"]

    of_val = _of(of_chi(ctx["nuTilda_of"], nu))
    nn_val = _nn(ctx["nuTilda"] / nu)

    _report("chi", of_val, nn_val)
    np.testing.assert_allclose(nn_val, of_val, rtol=1e-10, atol=0)


# ---------------------------------------------------------------------------
# Step 2: fv1
# ---------------------------------------------------------------------------


def test_fv1(turbulence_context: dict[str, Any]) -> None:
    ctx = turbulence_context
    nu = ctx["nu_value"]

    chi_of = of_chi(ctx["nuTilda_of"], nu)
    of_val = _of(of_fv1(chi_of, CFG_OF.Cv1))

    chi_nn = ctx["nuTilda"] / nu
    nn_val = _nn(chi_nn**3 / (chi_nn**3 + CFG_NN.Cv1**3))

    _report("fv1", of_val, nn_val)
    np.testing.assert_allclose(nn_val, of_val, rtol=1e-10, atol=1e-15)


# ---------------------------------------------------------------------------
# Step 3: fv2
# ---------------------------------------------------------------------------


def test_fv2(turbulence_context: dict[str, Any]) -> None:
    ctx = turbulence_context
    nu = ctx["nu_value"]

    chi_of = of_chi(ctx["nuTilda_of"], nu)
    fv1_of_val = of_fv1(chi_of, CFG_OF.Cv1)
    of_val = _of(of_fv2(chi_of, fv1_of_val))

    chi_nn = ctx["nuTilda"] / nu
    fv1_nn = chi_nn**3 / (chi_nn**3 + CFG_NN.Cv1**3)
    nn_val = _nn(1.0 - chi_nn / (1.0 + chi_nn * fv1_nn))

    _report("fv2", of_val, nn_val)
    np.testing.assert_allclose(nn_val, of_val, rtol=1e-10, atol=1e-15)


# ---------------------------------------------------------------------------
# Step 4: Omega
# ---------------------------------------------------------------------------


def test_Omega(turbulence_context: dict[str, Any]) -> None:
    ctx = turbulence_context

    of_val = _of(of_Omega(ctx["U_of"]))

    grad_U = nn.exp.grad_field(ctx["U"])
    nn_val = _nn(nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U)), 0.0))

    _report("Omega", of_val, nn_val)
    np.testing.assert_allclose(nn_val, of_val, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Step 5: Stilda
# ---------------------------------------------------------------------------


def test_Stilda(turbulence_context: dict[str, Any]) -> None:
    ctx = turbulence_context
    nu = ctx["nu_value"]
    mesh = ctx["U_of"].mesh()

    # pybFoam
    chi_of = of_chi(ctx["nuTilda_of"], nu)
    fv1_of_val = of_fv1(chi_of, CFG_OF.Cv1)
    fv2_of_val = of_fv2(chi_of, fv1_of_val)
    Omega_of = of_Omega(ctx["U_of"])
    d_of = wallDist.New(mesh).y()
    Stilda_of_val, _ = of_Stilda(Omega_of, fv2_of_val, ctx["nuTilda_of"], d_of, CFG_OF)
    of_val = _of(Stilda_of_val)

    # NeoN
    chi_nn = ctx["nuTilda"] / nu
    fv1_nn = chi_nn**3 / (chi_nn**3 + CFG_NN.Cv1**3)
    fv2_nn = 1.0 - chi_nn / (1.0 + chi_nn * fv1_nn)
    grad_U = nn.exp.grad_field(ctx["U"])
    Omega_nn = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U)), 0.0)
    d_safe = nn.field_max(ctx["d"], SMALL)
    Stilda_nn = nn.field_max(
        Omega_nn + fv2_nn * ctx["nuTilda"] / (CFG_NN.kappa**2 * d_safe**2),
        CFG_NN.Cs * Omega_nn,
    )
    nn_val = _nn(Stilda_nn)

    _report("Stilda", of_val, nn_val)
    np.testing.assert_allclose(nn_val, of_val, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Step 6: fw
# ---------------------------------------------------------------------------


def test_fw(turbulence_context: dict[str, Any]) -> None:
    ctx = turbulence_context
    nu = ctx["nu_value"]
    mesh = ctx["U_of"].mesh()

    # pybFoam
    chi_of = of_chi(ctx["nuTilda_of"], nu)
    fv1_of_val = of_fv1(chi_of, CFG_OF.Cv1)
    fv2_of_val = of_fv2(chi_of, fv1_of_val)
    Omega_of = of_Omega(ctx["U_of"])
    d_of = wallDist.New(mesh).y()
    Stilda_of_val, kd2_of = of_Stilda(Omega_of, fv2_of_val, ctx["nuTilda_of"], d_of, CFG_OF)
    fw_of = of_fw(ctx["nuTilda_of"], Stilda_of_val, kd2_of, CFG_OF)
    of_val = _of(fw_of)

    # NeoN
    chi_nn = ctx["nuTilda"] / nu
    fv1_nn = chi_nn**3 / (chi_nn**3 + CFG_NN.Cv1**3)
    fv2_nn = 1.0 - chi_nn / (1.0 + chi_nn * fv1_nn)
    grad_U = nn.exp.grad_field(ctx["U"])
    Omega_nn = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U)), 0.0)
    d_safe = nn.field_max(ctx["d"], SMALL)
    Stilda_nn = nn.field_max(
        Omega_nn + fv2_nn * ctx["nuTilda"] / (CFG_NN.kappa**2 * d_safe**2),
        CFG_NN.Cs * Omega_nn,
    )
    Stilda_safe = nn.field_max(Stilda_nn, SMALL)
    r_nn = nn.field_min(ctx["nuTilda"] / (Stilda_safe * CFG_NN.kappa**2 * d_safe**2), 10.0)
    g_nn = r_nn + CFG_NN.Cw2 * (r_nn**6 - r_nn)
    g6_nn = g_nn**6
    fw_frac = (1.0 + CFG_NN.Cw3**6) * (g6_nn + CFG_NN.Cw3**6) ** (-1.0)
    fw_nn = g_nn * nn.field_pow(fw_frac, 1.0 / 6.0)
    nn_val = _nn(fw_nn)

    _report("fw", of_val, nn_val)
    np.testing.assert_allclose(nn_val, of_val, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Step 7: Sn_coeff (destruction coefficient)
# ---------------------------------------------------------------------------


def test_Sn_coeff(turbulence_context: dict[str, Any]) -> None:
    ctx = turbulence_context
    nu = ctx["nu_value"]
    mesh = ctx["U_of"].mesh()

    # pybFoam
    chi_of = of_chi(ctx["nuTilda_of"], nu)
    fv1_of_val = of_fv1(chi_of, CFG_OF.Cv1)
    fv2_of_val = of_fv2(chi_of, fv1_of_val)
    Omega_of = of_Omega(ctx["U_of"])
    d_of = wallDist.New(mesh).y()
    Stilda_of_val, kd2_of = of_Stilda(Omega_of, fv2_of_val, ctx["nuTilda_of"], d_of, CFG_OF)
    fw_of = of_fw(ctx["nuTilda_of"], Stilda_of_val, kd2_of, CFG_OF)
    d2_of = volScalarField(pyf.sqr(d_of))
    Sn_of = volScalarField(CFG_OF.Cw1 * fw_of * ctx["nuTilda_of"] / d2_of)
    of_val = _of(Sn_of)

    # NeoN
    chi_nn = ctx["nuTilda"] / nu
    fv1_nn = chi_nn**3 / (chi_nn**3 + CFG_NN.Cv1**3)
    fv2_nn = 1.0 - chi_nn / (1.0 + chi_nn * fv1_nn)
    grad_U = nn.exp.grad_field(ctx["U"])
    Omega_nn = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U)), 0.0)
    d_safe = nn.field_max(ctx["d"], SMALL)
    Stilda_nn = nn.field_max(
        Omega_nn + fv2_nn * ctx["nuTilda"] / (CFG_NN.kappa**2 * d_safe**2),
        CFG_NN.Cs * Omega_nn,
    )
    Stilda_safe = nn.field_max(Stilda_nn, SMALL)
    r_nn = nn.field_min(ctx["nuTilda"] / (Stilda_safe * CFG_NN.kappa**2 * d_safe**2), 10.0)
    g_nn = r_nn + CFG_NN.Cw2 * (r_nn**6 - r_nn)
    g6_nn = g_nn**6
    fw_frac = (1.0 + CFG_NN.Cw3**6) * (g6_nn + CFG_NN.Cw3**6) ** (-1.0)
    fw_nn = g_nn * nn.field_pow(fw_frac, 1.0 / 6.0)
    Sn_nn = CFG_NN.Cw1 * fw_nn * ctx["nuTilda"] / d_safe**2
    nn_val = _nn(Sn_nn)

    _report("Sn_coeff", of_val, nn_val)
    np.testing.assert_allclose(nn_val, of_val, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Step 8: production
# ---------------------------------------------------------------------------


def test_production(turbulence_context: dict[str, Any]) -> None:
    ctx = turbulence_context
    nu = ctx["nu_value"]
    mesh = ctx["U_of"].mesh()

    # pybFoam
    chi_of = of_chi(ctx["nuTilda_of"], nu)
    fv1_of_val = of_fv1(chi_of, CFG_OF.Cv1)
    fv2_of_val = of_fv2(chi_of, fv1_of_val)
    Omega_of = of_Omega(ctx["U_of"])
    d_of = wallDist.New(mesh).y()
    Stilda_of_val, _ = of_Stilda(Omega_of, fv2_of_val, ctx["nuTilda_of"], d_of, CFG_OF)
    prod_of = volScalarField(CFG_OF.Cb1 * Stilda_of_val * ctx["nuTilda_of"])
    of_val = _of(prod_of)

    # NeoN
    chi_nn = ctx["nuTilda"] / nu
    fv1_nn = chi_nn**3 / (chi_nn**3 + CFG_NN.Cv1**3)
    fv2_nn = 1.0 - chi_nn / (1.0 + chi_nn * fv1_nn)
    grad_U = nn.exp.grad_field(ctx["U"])
    Omega_nn = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U)), 0.0)
    d_safe = nn.field_max(ctx["d"], SMALL)
    Stilda_nn = nn.field_max(
        Omega_nn + fv2_nn * ctx["nuTilda"] / (CFG_NN.kappa**2 * d_safe**2),
        CFG_NN.Cs * Omega_nn,
    )
    nn_val = _nn(CFG_NN.Cb1 * Stilda_nn * ctx["nuTilda"])

    _report("production", of_val, nn_val)
    np.testing.assert_allclose(nn_val, of_val, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Step 9: nonConsDiff
# ---------------------------------------------------------------------------


def test_nonConsDiff(turbulence_context: dict[str, Any]) -> None:
    ctx = turbulence_context

    # pybFoam
    grad_nT_of = volVectorField(fvc.grad(ctx["nuTilda_of"]))
    magSqr_of = volScalarField(pyf.magSqr(grad_nT_of))
    of_val = _of(magSqr_of) * (CFG_OF.Cb2 / CFG_OF.sigma)

    # NeoN
    grad_nT_nn = nn.exp.grad_field(ctx["nuTilda"])
    nn_val = _nn((CFG_NN.Cb2 / CFG_NN.sigma) * nn.inner(grad_nT_nn, grad_nT_nn))

    _report("nonConsDiff", of_val, nn_val)
    np.testing.assert_allclose(nn_val, of_val, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Step 10: DnuTildaEff
# ---------------------------------------------------------------------------


def test_DnuTildaEff(turbulence_context: dict[str, Any]) -> None:
    ctx = turbulence_context
    nu = ctx["nu_value"]

    # pybFoam
    of_val = _of(volScalarField(of_DnuTildaEff(ctx["nuTilda_of"], nu, CFG_OF.sigma)))

    # NeoN
    nn_val = _nn((nu + ctx["nuTilda"]) / CFG_NN.sigma)

    _report("DnuTildaEff", of_val, nn_val)
    np.testing.assert_allclose(nn_val, of_val, rtol=1e-10, atol=1e-15)


# ---------------------------------------------------------------------------
# Step 11: Full correct() — nuTilda result
# ---------------------------------------------------------------------------


def test_full_correct(turbulence_context: dict[str, Any]) -> None:
    """Full SA correct(): pybFoam (=OF) vs NeoN."""
    ctx = turbulence_context
    nu = ctx["nu_value"]

    # pybFoam: turbulence.correct() (the compiled reference)
    ctx["laminar_transport"].correct()
    ctx["turbulence"].correct()
    of_val = _of(ctx["nuTilda_of"])

    # NeoN
    from neofoam.solver.incompressibleFluidNeon.models.turbulence.spalartAllmaras import (
        correct as sa_correct_neon,
    )

    sa_correct_neon(
        CFG_NN, ctx["rt"],
        ctx["nuTilda"], ctx["nut"], ctx["U"], ctx["phi"], ctx["d"], nu,
    )
    nn_val = _nn(ctx["nuTilda"])

    _report("nuTilda after correct()", of_val, nn_val)

    np.testing.assert_allclose(
        nn_val, of_val, rtol=1e-3, atol=1e-10,
        err_msg="NeoN SA correct() does not match pybFoam/OF correct()",
    )
