# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Manual SA equation assembly: OF fvm vs NeoN imp.

Assembles progressively more complex equations using pybFoam's fvm operators
and NeoN's imp operators, comparing results to isolate the divergence source.
Uses the turbulence_context fixture fields directly (no re-reading).
"""

from typing import Any

import numpy as np
import neon._neon as nn
import pybFoam as pyf
from pybFoam import fvm, fvScalarMatrix, volScalarField

from neofoam import neofoam_bindings as nfb
from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    SpalartAllmarasConfig,
)


def _of_internal(field: Any) -> np.ndarray:
    return np.array(field.internalField())


def _nn_internal(field: Any) -> np.ndarray:
    return np.asarray(field.internal_vector().__array__())


def _report(name: str, a: np.ndarray, b: np.ndarray, la: str = "A", lb: str = "B") -> None:
    abs_diff = np.max(np.abs(a - b))
    denom = np.max(np.abs(a)) + 1e-30
    print(f"\n  {name}:")
    print(f"    {la}: min={a.min():.6e}  max={a.max():.6e}  mean={a.mean():.6e}")
    print(f"    {lb}: min={b.min():.6e}  max={b.max():.6e}  mean={b.mean():.6e}")
    print(f"    diff: abs={abs_diff:.6e}  rel={abs_diff / denom:.6e}")


# ---------------------------------------------------------------------------
# 1. ddt only: OF fvm::ddt vs NeoN nn.imp.ddt
#    With old=current, nuTilda should be unchanged after solve.
# ---------------------------------------------------------------------------


def test_ddt_only(turbulence_context: dict[str, Any]) -> None:
    """fvm::ddt(nuTilda) solved by OF: field should be unchanged."""
    ctx = turbulence_context

    nuTilda_of = ctx["nuTilda_of"]
    before = _of_internal(nuTilda_of).copy()

    eqn = fvScalarMatrix(fvm.ddt(nuTilda_of))
    eqn.solve()
    after = _of_internal(nuTilda_of)

    _report("ddt-only OF", before, after, "before", "after")
    change = np.max(np.abs(after - before))
    print(f"    max change: {change:.6e}")
    np.testing.assert_allclose(after, before, rtol=1e-8, atol=1e-15,
                               err_msg="ddt-only should preserve initial state")


# ---------------------------------------------------------------------------
# 2. ddt + div: OF vs NeoN
# ---------------------------------------------------------------------------


def test_ddt_div(turbulence_context: dict[str, Any]) -> None:
    """ddt + div(phi, nuTilda): OF fvm solve vs NeoN imp solve."""
    ctx = turbulence_context

    # OF solve
    nuTilda_of = ctx["nuTilda_of"]
    phi_of = pyf.createPhi(ctx["U_of"])
    before_of = _of_internal(nuTilda_of).copy()

    eqn_of = fvScalarMatrix(fvm.ddt(nuTilda_of) + fvm.div(phi_of, nuTilda_of))
    eqn_of.solve()
    after_of = _of_internal(nuTilda_of)

    # NeoN solve
    nuTilda_nn = ctx["nuTilda"]
    before_nn = _nn_internal(nuTilda_nn).copy()
    nn.rotate_old_times(nuTilda_nn)

    eqn_nn = nfb.PDESolverScalar(
        nn.imp.ddt(nuTilda_nn) + nn.imp.div(ctx["phi"], nuTilda_nn),
        nuTilda_nn, ctx["rt"],
    )
    eqn_nn.solve()
    after_nn = _nn_internal(nuTilda_nn)

    _report("ddt+div OF", before_of, after_of, "before", "after")
    _report("ddt+div NeoN", before_nn, after_nn, "before", "after")
    _report("ddt+div OF vs NeoN result", after_of, after_nn, "OF", "NeoN")

    np.testing.assert_allclose(after_nn, after_of, rtol=1e-3, atol=1e-10,
                               err_msg="ddt+div differs between OF and NeoN")


# ---------------------------------------------------------------------------
# 3. ddt + div - laplacian: OF vs NeoN
# ---------------------------------------------------------------------------


def test_ddt_div_laplacian(turbulence_context: dict[str, Any]) -> None:
    """ddt + div - laplacian: OF fvm solve vs NeoN imp solve."""
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]

    # OF solve
    nuTilda_of = ctx["nuTilda_of"]
    phi_of = pyf.createPhi(ctx["U_of"])
    # Use nuTilda/sigma as laplacian coefficient (avoids float+volScalarField issue)
    DnuTildaEff_of = nuTilda_of / cfg.sigma
    before_of = _of_internal(nuTilda_of).copy()

    eqn_of = fvScalarMatrix(
        fvm.ddt(nuTilda_of)
        + fvm.div(phi_of, nuTilda_of)
        - fvm.laplacian(DnuTildaEff_of, nuTilda_of)
    )
    eqn_of.solve()
    after_of = _of_internal(nuTilda_of)

    # NeoN solve
    nuTilda_nn = ctx["nuTilda"]
    before_nn = _nn_internal(nuTilda_nn).copy()
    nn.rotate_old_times(nuTilda_nn)
    # Same coefficient: nuTilda/sigma (matches OF side)
    DnuTildaEff_nn = nuTilda_nn / cfg.sigma
    interp = nn.SurfaceInterpolationScalar(
        ctx["rt"].executor, ctx["rt"].nf_mesh, nn.TokenList(["linear"])
    )
    DnuTildaEff_f = interp.interpolate(DnuTildaEff_nn)

    eqn_nn = nfb.PDESolverScalar(
        nn.imp.ddt(nuTilda_nn)
        + nn.imp.div(ctx["phi"], nuTilda_nn)
        - nn.imp.laplacian(DnuTildaEff_f, nuTilda_nn),
        nuTilda_nn, ctx["rt"],
    )
    eqn_nn.solve()
    after_nn = _nn_internal(nuTilda_nn)

    _report("transport OF", before_of, after_of, "before", "after")
    _report("transport NeoN", before_nn, after_nn, "before", "after")
    _report("transport OF vs NeoN", after_of, after_nn, "OF", "NeoN")

    np.testing.assert_allclose(after_nn, after_of, rtol=1e-3, atol=1e-10,
                               err_msg="Transport equation differs between OF and NeoN")


# ---------------------------------------------------------------------------
# 4. ddt + div - laplacian + Sp(destruction): OF vs NeoN
# ---------------------------------------------------------------------------


def test_with_implicit_source(turbulence_context: dict[str, Any]) -> None:
    """Transport + implicit destruction source: OF fvm.Sp vs NeoN imp.source."""
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]

    # --- Compute Sn_coeff as OF volScalarField ---
    nuTilda_of = ctx["nuTilda_of"]
    phi_of = pyf.createPhi(ctx["U_of"])
    DnuTildaEff_of = nuTilda_of / cfg.sigma

    # Use a simple Sp coefficient: dimensionedScalar constant
    Sn_simple = pyf.dimensionedScalar("Sn", pyf.dimless / pyf.dimTime, 100.0)

    before_of = _of_internal(nuTilda_of).copy()
    eqn_of = fvScalarMatrix(
        fvm.ddt(nuTilda_of)
        + fvm.div(phi_of, nuTilda_of)
        - fvm.laplacian(DnuTildaEff_of, nuTilda_of)
        + fvm.Sp(Sn_simple, nuTilda_of)
    )
    eqn_of.solve()
    after_of = _of_internal(nuTilda_of)

    # --- NeoN same equation ---
    nuTilda_nn = ctx["nuTilda"]
    before_nn = _nn_internal(nuTilda_nn).copy()
    nn.rotate_old_times(nuTilda_nn)
    DnuTildaEff_nn = nuTilda_nn / cfg.sigma
    interp = nn.SurfaceInterpolationScalar(
        ctx["rt"].executor, ctx["rt"].nf_mesh, nn.TokenList(["linear"])
    )
    DnuTildaEff_f = interp.interpolate(DnuTildaEff_nn)

    # Same constant: create a uniform scalar field with value 100
    Sn_nn = nn.ScalarVolumeField(ctx["rt"].executor, "Sn", ctx["rt"].nf_mesh)
    nn.fill(Sn_nn.internal_vector(), 100.0)

    eqn_nn = nfb.PDESolverScalar(
        nn.imp.ddt(nuTilda_nn)
        + nn.imp.div(ctx["phi"], nuTilda_nn)
        - nn.imp.laplacian(DnuTildaEff_f, nuTilda_nn)
        + nn.imp.source(Sn_nn, nuTilda_nn),
        nuTilda_nn, ctx["rt"],
    )
    eqn_nn.solve()
    after_nn = _nn_internal(nuTilda_nn)

    _report("with Sp: OF", before_of, after_of, "before", "after")
    _report("with Sp: NeoN", before_nn, after_nn, "before", "after")
    _report("with Sp: OF vs NeoN", after_of, after_nn, "OF", "NeoN")

    np.testing.assert_allclose(after_nn, after_of, rtol=1e-3, atol=1e-10,
                               err_msg="Implicit source differs between OF and NeoN")


# ---------------------------------------------------------------------------
# 5. ddt + div - laplacian - Su(explicit source): OF vs NeoN
# ---------------------------------------------------------------------------


def test_with_explicit_source(turbulence_context: dict[str, Any]) -> None:
    """Transport + explicit source: OF fvm.Su vs NeoN exp.source."""
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()

    nuTilda_of = ctx["nuTilda_of"]
    phi_of = pyf.createPhi(ctx["U_of"])
    DnuTildaEff_of = nuTilda_of / cfg.sigma

    # Explicit source: dimensionedScalar (adds constant/volume to RHS)
    rhs_of = pyf.dimensionedScalar("rhs", pyf.dimViscosity / pyf.dimTime, 50.0)

    before_of = _of_internal(nuTilda_of).copy()
    eqn_of = fvScalarMatrix(
        fvm.ddt(nuTilda_of)
        + fvm.div(phi_of, nuTilda_of)
        - fvm.laplacian(DnuTildaEff_of, nuTilda_of)
        - fvm.Su(rhs_of, nuTilda_of)
    )
    eqn_of.solve()
    after_of = _of_internal(nuTilda_of)

    # NeoN
    nuTilda_nn = ctx["nuTilda"]
    before_nn = _nn_internal(nuTilda_nn).copy()
    nn.rotate_old_times(nuTilda_nn)
    DnuTildaEff_nn = nuTilda_nn / cfg.sigma
    interp = nn.SurfaceInterpolationScalar(
        ctx["rt"].executor, ctx["rt"].nf_mesh, nn.TokenList(["linear"])
    )
    DnuTildaEff_f = interp.interpolate(DnuTildaEff_nn)

    # Same constant explicit source
    rhs_nn = nn.ScalarVolumeField(ctx["rt"].executor, "rhs", ctx["rt"].nf_mesh)
    nn.fill(rhs_nn.internal_vector(), 50.0)
    ones = nn.ScalarVolumeField(ctx["rt"].executor, "ones", ctx["rt"].nf_mesh)
    nn.fill(ones.internal_vector(), 1.0)

    eqn_nn = nfb.PDESolverScalar(
        nn.imp.ddt(nuTilda_nn)
        + nn.imp.div(ctx["phi"], nuTilda_nn)
        - nn.imp.laplacian(DnuTildaEff_f, nuTilda_nn)
        - nn.exp.source(rhs_nn, ones),
        nuTilda_nn, ctx["rt"],
    )
    eqn_nn.solve()
    after_nn = _nn_internal(nuTilda_nn)

    _report("with Su: OF", before_of, after_of, "before", "after")
    _report("with Su: NeoN", before_nn, after_nn, "before", "after")
    _report("with Su: OF vs NeoN", after_of, after_nn, "OF", "NeoN")

    np.testing.assert_allclose(after_nn, after_of, rtol=1e-3, atol=1e-10,
                               err_msg="Explicit source differs between OF and NeoN")


# ---------------------------------------------------------------------------
# 6. Full equation: ddt + div - laplacian + Sp - Su (both sources)
# ---------------------------------------------------------------------------


def test_with_both_sources(turbulence_context: dict[str, Any]) -> None:
    """Transport + both sources: OF vs NeoN."""
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()

    nuTilda_of = ctx["nuTilda_of"]
    phi_of = pyf.createPhi(ctx["U_of"])
    DnuTildaEff_of = nuTilda_of / cfg.sigma

    Sn_of = pyf.dimensionedScalar("Sn", pyf.dimless / pyf.dimTime, 100.0)
    rhs_of = pyf.dimensionedScalar("rhs", pyf.dimViscosity / pyf.dimTime, 50.0)

    before_of = _of_internal(nuTilda_of).copy()
    eqn_of = fvScalarMatrix(
        fvm.ddt(nuTilda_of)
        + fvm.div(phi_of, nuTilda_of)
        - fvm.laplacian(DnuTildaEff_of, nuTilda_of)
        + fvm.Sp(Sn_of, nuTilda_of)
        - fvm.Su(rhs_of, nuTilda_of)
    )
    eqn_of.solve()
    after_of = _of_internal(nuTilda_of)

    # NeoN
    nuTilda_nn = ctx["nuTilda"]
    before_nn = _nn_internal(nuTilda_nn).copy()
    nn.rotate_old_times(nuTilda_nn)
    DnuTildaEff_nn = nuTilda_nn / cfg.sigma
    interp = nn.SurfaceInterpolationScalar(
        ctx["rt"].executor, ctx["rt"].nf_mesh, nn.TokenList(["linear"])
    )
    DnuTildaEff_f = interp.interpolate(DnuTildaEff_nn)

    Sn_nn = nn.ScalarVolumeField(ctx["rt"].executor, "Sn2", ctx["rt"].nf_mesh)
    nn.fill(Sn_nn.internal_vector(), 100.0)
    rhs_nn = nn.ScalarVolumeField(ctx["rt"].executor, "rhs2", ctx["rt"].nf_mesh)
    nn.fill(rhs_nn.internal_vector(), 50.0)
    ones = nn.ScalarVolumeField(ctx["rt"].executor, "ones2", ctx["rt"].nf_mesh)
    nn.fill(ones.internal_vector(), 1.0)

    eqn_nn = nfb.PDESolverScalar(
        nn.imp.ddt(nuTilda_nn)
        + nn.imp.div(ctx["phi"], nuTilda_nn)
        - nn.imp.laplacian(DnuTildaEff_f, nuTilda_nn)
        + nn.imp.source(Sn_nn, nuTilda_nn)
        - nn.exp.source(rhs_nn, ones),
        nuTilda_nn, ctx["rt"],
    )
    eqn_nn.solve()
    after_nn = _nn_internal(nuTilda_nn)

    _report("both sources: OF", before_of, after_of, "before", "after")
    _report("both sources: NeoN", before_nn, after_nn, "before", "after")
    _report("both sources: OF vs NeoN", after_of, after_nn, "OF", "NeoN")

    np.testing.assert_allclose(after_nn, after_of, rtol=1e-3, atol=1e-10,
                               err_msg="Both sources differ between OF and NeoN")


# ---------------------------------------------------------------------------
# 7. Field-valued Sp: ddt + div - laplacian + Sp(nuTilda/d², nuTilda)
#    This is the key test — constant Sp matches, does field Sp?
# ---------------------------------------------------------------------------


def test_field_valued_sp(turbulence_context: dict[str, Any]) -> None:
    """Transport + field-valued Sp(nuTilda/d², nuTilda): OF vs NeoN.

    Uses the actual SA destruction coefficient pattern (nuTilda/d²) as
    the Sp coefficient. This is the simplest test that reproduces the
    mismatch seen in the full SA comparison.
    """
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    mesh = ctx["U_of"].mesh()

    # --- OF: compute Sn_coeff as volScalarField ---
    nuTilda_of = ctx["nuTilda_of"]
    phi_of = pyf.createPhi(ctx["U_of"])
    DnuTildaEff_of = nuTilda_of / cfg.sigma

    # d as volScalarField from wallDist
    from pybFoam import wallDist
    d_of = wallDist.New(mesh).y()

    # Sn_coeff = 1.5 * nuTilda / d²  (Cw1=1.5 when Cb1=Cb2=0)
    Sn_of_tmp = 1.5 * nuTilda_of / (d_of * d_of)
    Sn_of = volScalarField(Sn_of_tmp)  # materialized copy for inspection

    before_of = _of_internal(nuTilda_of).copy()
    eqn_of = fvScalarMatrix(
        fvm.ddt(nuTilda_of)
        + fvm.div(phi_of, nuTilda_of)
        - fvm.laplacian(DnuTildaEff_of, nuTilda_of)
        + fvm.Sp(1.5 * nuTilda_of / (d_of * d_of), nuTilda_of)
    )
    eqn_of.solve()
    after_of = _of_internal(nuTilda_of)

    # --- NeoN: same equation ---
    nuTilda_nn = ctx["nuTilda"]
    before_nn = _nn_internal(nuTilda_nn).copy()
    nn.rotate_old_times(nuTilda_nn)

    DnuTildaEff_nn = nuTilda_nn / cfg.sigma
    interp = nn.SurfaceInterpolationScalar(
        ctx["rt"].executor, ctx["rt"].nf_mesh, nn.TokenList(["linear"])
    )
    DnuTildaEff_f = interp.interpolate(DnuTildaEff_nn)

    Sn_nn = 1.5 * nuTilda_nn / (ctx["d"] * ctx["d"])

    # Verify Sp coefficients match before solve
    sn_of_arr = _of_internal(Sn_of)
    sn_nn_arr = _nn_internal(Sn_nn)
    _report("Sn_coeff (before solve)", sn_of_arr, sn_nn_arr)

    eqn_nn = nfb.PDESolverScalar(
        nn.imp.ddt(nuTilda_nn)
        + nn.imp.div(ctx["phi"], nuTilda_nn)
        - nn.imp.laplacian(DnuTildaEff_f, nuTilda_nn)
        + nn.imp.source(Sn_nn, nuTilda_nn),
        nuTilda_nn, ctx["rt"],
    )
    eqn_nn.solve()
    after_nn = _nn_internal(nuTilda_nn)

    _report("field Sp: OF", before_of, after_of, "before", "after")
    _report("field Sp: NeoN", before_nn, after_nn, "before", "after")
    _report("field Sp: OF vs NeoN", after_of, after_nn, "OF", "NeoN")

    np.testing.assert_allclose(after_nn, after_of, rtol=1e-3, atol=1e-10,
                               err_msg="Field-valued Sp differs between OF and NeoN")


# ---------------------------------------------------------------------------
# 8. Full SA via pybFoam fvm: reproduce turbulence.correct() manually
# ---------------------------------------------------------------------------


def test_relax_effect(turbulence_context: dict[str, Any]) -> None:
    """Compare OF fvm assembly WITH vs WITHOUT relax().

    Uses the same field-valued Sp coefficient. If relax changes the result
    significantly, that explains the mismatch with NeoN (which doesn't relax).
    """
    ctx = turbulence_context
    cfg = SpalartAllmarasConfig()
    nu = ctx["nu_value"]

    nuTilda_of = ctx["nuTilda_of"]
    phi_of = pyf.createPhi(ctx["U_of"])

    from pybFoam import wallDist
    d_of = wallDist.New(ctx["U_of"].mesh()).y()
    DnuTildaEff_of = (nuTilda_of + nu) / cfg.sigma
    Sn_of = cfg.Cw1 * nuTilda_of / (d_of * d_of)

    # --- Solve WITHOUT relax ---
    before = _of_internal(nuTilda_of).copy()
    eqn_no_relax = fvScalarMatrix(
        fvm.ddt(nuTilda_of)
        + fvm.div(phi_of, nuTilda_of)
        - fvm.laplacian(DnuTildaEff_of, nuTilda_of)
        + fvm.Sp(Sn_of, nuTilda_of)
    )
    eqn_no_relax.solve()
    after_no_relax = _of_internal(nuTilda_of).copy()

    # --- Solve WITH relax (need fresh nuTilda — but we can't re-read) ---
    # Instead, compare no-relax OF vs NeoN
    nuTilda_nn = ctx["nuTilda"]
    nn.rotate_old_times(nuTilda_nn)

    DnuTildaEff_nn = (nu + nuTilda_nn) / cfg.sigma
    interp = nn.SurfaceInterpolationScalar(
        ctx["rt"].executor, ctx["rt"].nf_mesh, nn.TokenList(["linear"])
    )
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
    after_nn = _nn_internal(nuTilda_nn)

    _report("OF no-relax vs NeoN (field Sp)", after_no_relax, after_nn, "OF", "NeoN")

    # This should match since test_field_valued_sp already passed
    np.testing.assert_allclose(after_nn, after_no_relax, rtol=1e-3, atol=1e-10,
                               err_msg="OF (no relax) vs NeoN with field Sp")
