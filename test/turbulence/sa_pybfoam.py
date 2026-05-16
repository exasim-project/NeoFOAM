# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spalart-Allmaras model implemented with pybFoam fvm operators.

Mirrors SpalartAllmarasBase::correct() from OpenFOAM v2406.
Validated to match turbulence.correct() to rel diff ~3e-6.

Function signatures mirror the NeoN implementation in
src/neofoam/solver/incompressibleFluidNeon/models/turbulence/spalartAllmaras.py
for easy comparison.
"""

from dataclasses import dataclass
from typing import Any

import pybFoam as pyf
from pybFoam import (
    fvc,
    fvm,
    fvScalarMatrix,
    volScalarField,
    volTensorField,
    wallDist,
)


@dataclass
class SAConstants:
    """SA model constants — matches SpalartAllmarasConfig."""

    sigma: float = 2.0 / 3.0
    Cb1: float = 0.1355
    Cb2: float = 0.622
    Cw2: float = 0.3
    Cw3: float = 2.0
    Cv1: float = 7.1
    kappa: float = 0.41
    Cs: float = 0.3

    @property
    def Cw1(self) -> float:
        return self.Cb1 / self.kappa**2 + (1 + self.Cb2) / self.sigma


def compute_chi(nuTilda: Any, nu: Any) -> Any:
    """chi = nuTilda / nu"""
    return volScalarField(nuTilda / nu)


def compute_fv1(chi: Any, Cv1: float) -> Any:
    """fv1 = chi³ / (chi³ + Cv1³)"""
    chi3 = volScalarField(pyf.pow3(chi))
    return volScalarField(chi3 / (chi3 + Cv1**3))


def compute_fv2(chi: Any, fv1: Any) -> Any:
    """fv2 = 1 - chi / (1 + chi*fv1)"""
    chi_fv1 = volScalarField(chi * fv1)
    denom = volScalarField(chi_fv1 + 1.0)
    ratio = volScalarField(chi / denom)
    return volScalarField(-ratio + 1.0)


def compute_Omega(U: Any) -> Any:
    """Omega = sqrt(2) * mag(skew(grad(U)))"""
    gradU = volTensorField(fvc.grad(U))
    return volScalarField(pyf.sqrt(2.0 * pyf.magSqr(pyf.skew(gradU))))


def compute_Stilda(
    Omega: Any, fv2: Any, nuTilda: Any, d: Any, cfg: SAConstants,
) -> Any:
    """Stilda = max(Omega + fv2*nuTilda/(kappa²*d²), Cs*Omega)"""
    kd2 = volScalarField(pyf.sqr(cfg.kappa * d))
    fv2_term = volScalarField(fv2 * nuTilda / kd2)
    arg1 = volScalarField(Omega + fv2_term)
    arg2 = volScalarField(cfg.Cs * Omega)
    return volScalarField(pyf.max(arg1, arg2)), kd2


def compute_fw(nuTilda: Any, Stilda: Any, kd2: Any, cfg: SAConstants) -> Any:
    """fw wall-damping function."""
    Ss = volScalarField(pyf.max(Stilda, 1e-10))
    r_denom = volScalarField(Ss * kd2)
    r = volScalarField(pyf.min(volScalarField(nuTilda / r_denom), 10.0))
    r6 = volScalarField(pyf.pow6(r))
    r6_minus_r = volScalarField(r6 - r)
    g = volScalarField(r + cfg.Cw2 * r6_minus_r)
    g6 = volScalarField(pyf.pow6(g))
    fw_denom = volScalarField(g6 + cfg.Cw3**6)
    fw_frac = volScalarField((1.0 + cfg.Cw3**6) / fw_denom)
    fw_pow = volScalarField(pyf.pow(fw_frac, 1.0 / 6.0))
    return volScalarField(g * fw_pow)


def compute_DnuTildaEff(nuTilda: Any, nu: Any, sigma: float) -> Any:
    """DnuTildaEff = (nuTilda + nu) / sigma"""
    return (nuTilda + nu) / sigma


def correct(
    nuTilda: Any,
    turb: Any,
    cfg: SAConstants | None = None,
) -> None:
    """Solve SA transport equation — mirrors SpalartAllmarasBase::correct().

    Args:
        nuTilda: volScalarField (from turb model registry)
        turb: incompressibleTurbulenceModel
        cfg: SA constants (defaults to standard values)
    """
    if cfg is None:
        cfg = SAConstants()

    nu = volScalarField(turb.nu())
    U = turb.U()
    phi = turb.phi()
    mesh = nuTilda.mesh()
    d = wallDist.New(mesh).y()

    # Auxiliary fields
    chi = compute_chi(nuTilda, nu)
    fv1 = compute_fv1(chi, cfg.Cv1)
    fv2 = compute_fv2(chi, fv1)
    Omega = compute_Omega(U)
    Stilda, kd2 = compute_Stilda(Omega, fv2, nuTilda, d, cfg)
    fw = compute_fw(nuTilda, Stilda, kd2, cfg)

    # Diffusion coefficient
    DnuTildaEff = compute_DnuTildaEff(nuTilda, nu, cfg.sigma)

    # Source terms
    d2 = volScalarField(pyf.sqr(d))
    production = cfg.Cb1 * Stilda * nuTilda
    Sn_coeff = cfg.Cw1 * fw * nuTilda / d2
    nonConsDiff_coeff = cfg.Cb2 / cfg.sigma

    # Equation assembly — matches OF's:
    #   ddt + div - laplacian - nonConsDiff == production - Sp(Sn, nuTilda)
    eqn = fvScalarMatrix(
        fvm.ddt(nuTilda)
        + fvm.div(phi, nuTilda)
        - fvm.laplacian(DnuTildaEff, nuTilda)
        - nonConsDiff_coeff * pyf.magSqr(fvc.grad(nuTilda))
        + fvm.Sp(Sn_coeff, nuTilda)
        - fvm.Su(production, nuTilda)
    )
    eqn.relax()
    eqn.solve()

    # Post-solve
    pyf.bound(nuTilda, pyf.dimensionedScalar("z", pyf.dimViscosity, 0.0))
    nuTilda.correctBoundaryConditions()
