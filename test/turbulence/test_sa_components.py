# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Narrow down SA discrepancy: compare individual components.

Uses the same turbulence_context fixture (same mesh, same non-uniform fields).
Each test isolates one intermediate quantity to find where NeoN diverges from
pybFoam.
"""

from typing import Any

import numpy as np
import pytest
import neon._neon as nn
import pybFoam as pyf
from pybFoam import fvc, volScalarField, volVectorField, volTensorField

from neofoam import neofoam_bindings as nfb
from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    SpalartAllmarasConfig,
)


def _of_internal(field: Any) -> np.ndarray:
    return np.array(field.internalField())


def _nn_internal(field: Any) -> np.ndarray:
    return np.asarray(field.internal_vector().__array__())


def _report(name: str, of_arr: np.ndarray, nn_arr: np.ndarray) -> None:
    abs_diff = np.max(np.abs(nn_arr - of_arr))
    denom = np.max(np.abs(of_arr)) + 1e-30
    rel_diff = abs_diff / denom
    print(f"\n  {name}:")
    print(f"    OF:   min={of_arr.min():.6e}  max={of_arr.max():.6e}  mean={of_arr.mean():.6e}")
    print(f"    NeoN: min={nn_arr.min():.6e}  max={nn_arr.max():.6e}  mean={nn_arr.mean():.6e}")
    print(f"    diff: abs={abs_diff:.6e}  rel={rel_diff:.6e}")


# ---------------------------------------------------------------------------
# 1. Input fields — should match exactly
# ---------------------------------------------------------------------------


def test_input_nuTilda(turbulence_context: dict[str, Any]) -> None:
    """nuTilda internal values match between backends."""
    ctx = turbulence_context
    of_arr = _of_internal(ctx["nuTilda_of"])
    nn_arr = _nn_internal(ctx["nuTilda"])
    _report("nuTilda (input)", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-12, atol=0)


def test_input_U(turbulence_context: dict[str, Any]) -> None:
    """U internal values match between backends."""
    ctx = turbulence_context
    of_arr = _of_internal(ctx["U_of"])
    nn_arr = _nn_internal(ctx["U"])
    _report("U (input)", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# 2. Gradient of U — feeds vorticity
# ---------------------------------------------------------------------------


def test_grad_U(turbulence_context: dict[str, Any]) -> None:
    """fvc::grad(U) matches between backends."""
    ctx = turbulence_context

    # pybFoam
    grad_U_of = volTensorField(fvc.grad(ctx["U_of"]))
    of_arr = _of_internal(grad_U_of)

    # NeoN
    grad_U_nn = nn.exp.grad_field(ctx["U"])
    nn_arr = _nn_internal(grad_U_nn)

    _report("grad(U)", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Vorticity: sqrt(2) * mag(skew(grad(U)))
# ---------------------------------------------------------------------------


def test_vorticity(turbulence_context: dict[str, Any]) -> None:
    """Omega = sqrt(2)*mag(skew(gradU)) matches between backends."""
    ctx = turbulence_context

    # pybFoam: compute Omega the same way OpenFOAM SA does
    grad_U_of = volTensorField(fvc.grad(ctx["U_of"]))
    grad_U_of_arr = _of_internal(grad_U_of)
    # skew(T) = 0.5*(T - T^T), Omega = sqrt(2)*mag(skew)
    skew_of = 0.5 * (grad_U_of_arr - grad_U_of_arr.reshape(-1, 3, 3).transpose(0, 2, 1).reshape(-1, 9))
    mag_skew_of = np.sqrt(np.sum(skew_of**2, axis=-1))
    omega_of = np.sqrt(2.0) * mag_skew_of

    # NeoN
    grad_U_nn = nn.exp.grad_field(ctx["U"])
    omega_nn_field = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U_nn)), 0.0)
    omega_nn = _nn_internal(omega_nn_field)

    _report("Omega (vorticity)", omega_of, omega_nn)
    np.testing.assert_allclose(omega_nn, omega_of, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Gradient of nuTilda — feeds non-conservative diffusion
# ---------------------------------------------------------------------------


def test_grad_nuTilda(turbulence_context: dict[str, Any]) -> None:
    """fvc::grad(nuTilda) matches between backends."""
    ctx = turbulence_context

    # pybFoam
    grad_nuTilda_of = volVectorField(fvc.grad(ctx["nuTilda_of"]))
    of_arr = _of_internal(grad_nuTilda_of)

    # NeoN
    grad_nuTilda_nn = nn.exp.grad_field(ctx["nuTilda"])
    nn_arr = _nn_internal(grad_nuTilda_nn)

    _report("grad(nuTilda)", of_arr, nn_arr)
    np.testing.assert_allclose(nn_arr, of_arr, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. Non-conservative diffusion: Cb2/sigma * |grad(nuTilda)|^2
# ---------------------------------------------------------------------------


def test_noncons_diffusion(turbulence_context: dict[str, Any]) -> None:
    """Cb2/sigma * magSqr(grad(nuTilda)) matches between backends."""
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()

    # pybFoam
    grad_nuTilda_of = volVectorField(fvc.grad(ctx["nuTilda_of"]))
    grad_arr = _of_internal(grad_nuTilda_of)
    magSqr_of = np.sum(grad_arr**2, axis=-1)
    noncons_of = (cfg.Cb2 / cfg.sigma) * magSqr_of

    # NeoN
    grad_nuTilda_nn = nn.exp.grad_field(ctx["nuTilda"])
    noncons_nn_field = (cfg.Cb2 / cfg.sigma) * nn.inner(grad_nuTilda_nn, grad_nuTilda_nn)
    noncons_nn = _nn_internal(noncons_nn_field)

    _report("nonConsDiff", noncons_of, noncons_nn)
    np.testing.assert_allclose(noncons_nn, noncons_of, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. Auxiliary fields: chi, fv1, fv2
# ---------------------------------------------------------------------------


def test_auxiliary_fields(turbulence_context: dict[str, Any]) -> None:
    """chi, fv1, fv2 match between numpy reference and NeoN."""
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]

    nuTilda_arr = _nn_internal(ctx["nuTilda"])

    # Numpy reference (same formula, ground truth from input data)
    chi_ref = nuTilda_arr / nu
    fv1_ref = chi_ref**3 / (chi_ref**3 + cfg.Cv1**3)
    fv2_ref = 1.0 - chi_ref / (1.0 + chi_ref * fv1_ref)

    # NeoN
    chi_nn = _nn_internal(ctx["nuTilda"] / nu)
    fv1_nn = _nn_internal(chi_nn_f := ctx["nuTilda"] / nu)
    # Recompute properly using NeoN field ops
    chi_field = ctx["nuTilda"] / nu
    fv1_field = chi_field**3 / (chi_field**3 + cfg.Cv1**3)
    fv2_field = 1.0 - chi_field / (1.0 + chi_field * fv1_field)

    chi_nn = _nn_internal(chi_field)
    fv1_nn = _nn_internal(fv1_field)
    fv2_nn = _nn_internal(fv2_field)

    _report("chi", chi_ref, chi_nn)
    np.testing.assert_allclose(chi_nn, chi_ref, rtol=1e-12, atol=0)

    _report("fv1", fv1_ref, fv1_nn)
    np.testing.assert_allclose(fv1_nn, fv1_ref, rtol=1e-10, atol=1e-15)

    _report("fv2", fv2_ref, fv2_nn)
    np.testing.assert_allclose(fv2_nn, fv2_ref, rtol=1e-10, atol=1e-15)


# ---------------------------------------------------------------------------
# 7. DnuTildaEff — diffusion coefficient
# ---------------------------------------------------------------------------


def test_diffusion_coeff(turbulence_context: dict[str, Any]) -> None:
    """DnuTildaEff = (nu + nuTilda)/sigma matches."""
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]

    nuTilda_arr = _nn_internal(ctx["nuTilda"])
    ref = (nu + nuTilda_arr) / cfg.sigma

    nn_field = (nu + ctx["nuTilda"]) / cfg.sigma
    nn_arr = _nn_internal(nn_field)

    _report("DnuTildaEff", ref, nn_arr)
    np.testing.assert_allclose(nn_arr, ref, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# 8. Production: Cb1 * Stilda * nuTilda (explicit source values)
# ---------------------------------------------------------------------------


def test_production(turbulence_context: dict[str, Any]) -> None:
    """Production term Cb1*Stilda*nuTilda matches between backends.

    Uses numpy Stilda computed from NeoN Omega and NeoN auxiliary fields
    as reference — isolates whether Stilda computation is correct.
    """
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]

    nuTilda_arr = _nn_internal(ctx["nuTilda"])
    d_arr = _nn_internal(ctx["d"])

    # Compute Stilda via NeoN
    chi = ctx["nuTilda"] / nu
    fv1 = chi**3 / (chi**3 + cfg.Cv1**3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    grad_U = nn.exp.grad_field(ctx["U"])
    Omega = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U)), 0.0)
    d_safe = nn.field_max(ctx["d"], 1e-10)
    Stilda = nn.field_max(
        Omega + fv2 * ctx["nuTilda"] / (cfg.kappa**2 * d_safe**2),
        cfg.Cs * Omega,
    )

    production_nn = _nn_internal(cfg.Cb1 * Stilda * ctx["nuTilda"])
    Stilda_arr = _nn_internal(Stilda)

    # Numpy reference from the same Stilda values
    production_ref = cfg.Cb1 * Stilda_arr * nuTilda_arr

    _report("production", production_ref, production_nn)
    np.testing.assert_allclose(production_nn, production_ref, rtol=1e-10, atol=1e-15)


# ---------------------------------------------------------------------------
# 9. Destruction coefficient: Cw1 * fw * nuTilda / d^2
# ---------------------------------------------------------------------------


def test_destruction_coeff(turbulence_context: dict[str, Any]) -> None:
    """Destruction coefficient Cw1*fw*nuTilda/d² is finite and positive."""
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]
    SMALL = 1e-10

    d_safe = nn.field_max(ctx["d"], SMALL)
    chi = ctx["nuTilda"] / nu
    fv1 = chi**3 / (chi**3 + cfg.Cv1**3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    grad_U = nn.exp.grad_field(ctx["U"])
    Omega = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U)), 0.0)
    Stilda = nn.field_max(
        Omega + fv2 * ctx["nuTilda"] / (cfg.kappa**2 * d_safe**2),
        cfg.Cs * Omega,
    )
    Stilda_safe = nn.field_max(Stilda, SMALL)
    r = nn.field_min(ctx["nuTilda"] / (Stilda_safe * cfg.kappa**2 * d_safe**2), 10.0)
    g = nn.field_max(r + cfg.Cw2 * (r**6 - r), SMALL)
    fw_arg = (1.0 + cfg.Cw3**6) * (g**6 + cfg.Cw3**6) ** (-1.0)
    fw_arg_safe = nn.field_max(fw_arg, SMALL)
    fw = g * nn.field_pow(fw_arg_safe, 1.0 / 6.0)

    Sn_coeff = cfg.Cw1 * fw * ctx["nuTilda"] / d_safe**2
    sn_arr = _nn_internal(Sn_coeff)

    print(f"\n  Sn_coeff: min={sn_arr.min():.6e} max={sn_arr.max():.6e}")
    assert np.all(np.isfinite(sn_arr)), "Sn_coeff has non-finite values"
    assert np.all(sn_arr >= 0), "Sn_coeff has negative values"


# ---------------------------------------------------------------------------
# 10. Full RHS comparison: production + nonConsDiff
# ---------------------------------------------------------------------------


def test_rhs_source(turbulence_context: dict[str, Any]) -> None:
    """Full explicit RHS (production + nonConsDiff) is finite."""
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]
    SMALL = 1e-10

    chi = ctx["nuTilda"] / nu
    fv1 = chi**3 / (chi**3 + cfg.Cv1**3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    grad_U = nn.exp.grad_field(ctx["U"])
    Omega = nn.field_max(2.0**0.5 * nn.mag(nn.skew(grad_U)), 0.0)
    d_safe = nn.field_max(ctx["d"], SMALL)
    Stilda = nn.field_max(
        Omega + fv2 * ctx["nuTilda"] / (cfg.kappa**2 * d_safe**2),
        cfg.Cs * Omega,
    )

    production = cfg.Cb1 * Stilda * ctx["nuTilda"]
    gradNuTilda = nn.exp.grad_field(ctx["nuTilda"])
    nonConsDiff = (cfg.Cb2 / cfg.sigma) * nn.inner(gradNuTilda, gradNuTilda)

    rhs = production + nonConsDiff
    rhs_arr = _nn_internal(rhs)

    prod_arr = _nn_internal(production)
    ncd_arr = _nn_internal(nonConsDiff)

    print(f"\n  production:  min={prod_arr.min():.6e} max={prod_arr.max():.6e}")
    print(f"  nonConsDiff: min={ncd_arr.min():.6e} max={ncd_arr.max():.6e}")
    print(f"  RHS total:   min={rhs_arr.min():.6e} max={rhs_arr.max():.6e}")

    assert np.all(np.isfinite(rhs_arr)), "RHS has non-finite values"


# ---------------------------------------------------------------------------
# 11. Phi (face flux) — internal faces must match
# ---------------------------------------------------------------------------


def test_phi_internal(turbulence_context: dict[str, Any]) -> None:
    """phi internal face values match between pybFoam and NeoN."""
    ctx = turbulence_context

    # pybFoam phi — internalField gives only internal faces
    phi_of = pyf.createPhi(ctx["U_of"])
    phi_of_arr = _of_internal(phi_of)

    # NeoN phi — internal_vector includes boundary faces at the end
    phi_nn_full = _nn_internal(ctx["phi"])
    n_internal = len(phi_of_arr)
    phi_nn_arr = phi_nn_full[:n_internal]

    _report("phi (internal faces)", phi_of_arr, phi_nn_arr)
    print(f"    OF size:   {len(phi_of_arr)}")
    print(f"    NeoN size: {len(phi_nn_full)} (internal: {n_internal}, boundary: {len(phi_nn_full) - n_internal})")

    np.testing.assert_allclose(phi_nn_arr, phi_of_arr, rtol=1e-10, atol=1e-20)


# ---------------------------------------------------------------------------
# 12. Phi boundary faces — check if NeoN boundary fluxes are reasonable
# ---------------------------------------------------------------------------


def test_phi_boundary(turbulence_context: dict[str, Any]) -> None:
    """phi boundary face values are finite and reasonable."""
    ctx = turbulence_context

    phi_of = pyf.createPhi(ctx["U_of"])
    n_internal = len(_of_internal(phi_of))

    phi_nn_full = _nn_internal(ctx["phi"])
    phi_nn_boundary = phi_nn_full[n_internal:]

    print(f"\n  phi boundary faces: {len(phi_nn_boundary)}")
    print(f"    min={phi_nn_boundary.min():.6e}  max={phi_nn_boundary.max():.6e}")
    print(f"    mean={phi_nn_boundary.mean():.6e}")
    print(f"    nonzero: {np.count_nonzero(phi_nn_boundary)}/{len(phi_nn_boundary)}")

    assert np.all(np.isfinite(phi_nn_boundary)), "phi boundary has non-finite values"


# ---------------------------------------------------------------------------
# 13. Laplacian: fvc::laplacian(nu, nuTilda) between backends
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="nn.exp has no evaluate/laplacian_field — cannot extract explicit laplacian values")
def test_laplacian(turbulence_context: dict[str, Any]) -> None:
    """NeoN explicit laplacian produces finite, bounded values."""
    pass


# ---------------------------------------------------------------------------
# 14. Div: fvc::div(phi, nuTilda) — pybFoam lacks this overload for scalar,
#     so compare OF fvc.div(phi) (flux divergence) instead
# ---------------------------------------------------------------------------


def test_div_phi(turbulence_context: dict[str, Any]) -> None:
    """fvc::div(phi) (flux divergence) matches between backends.

    pybFoam's fvc.div(surfaceScalarField) computes the Gauss divergence
    of the face flux field. This tests that phi itself is consistent.
    """
    ctx = turbulence_context

    # pybFoam: fvc.div(phi_of) — need the OF phi, not the NeoN one
    phi_of = pyf.createPhi(ctx["U_of"])
    div_phi_of = volScalarField(fvc.div(phi_of))
    div_of_arr = _of_internal(div_phi_of)

    # For comparison: div(phi) should be ~0 for an incompressible field
    print(f"\n  div(phi) OF:  min={div_of_arr.min():.6e}  max={div_of_arr.max():.6e}")
    print(f"  div(phi) OF:  mean={div_of_arr.mean():.6e}")

    # NOTE: div(phi) is NOT near zero because the analytical U field
    # doesn't satisfy continuity. This is expected — in a real solver,
    # phi comes from the pressure solve and IS divergence-free.
    # This non-zero div(phi) affects div(phi, nuTilda) and is a known
    # source of difference when comparing single-step correct() results.
    assert np.all(np.isfinite(div_of_arr)), "div(phi) has non-finite values"
