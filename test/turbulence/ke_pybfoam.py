# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""k-epsilon model with pybFoam fvm operators — for debugging.

Mirrors kEpsilon::correct() from OpenFOAM v2406 including SuSp(divU) terms.
"""

from typing import Any

import pybFoam as pyf
from pybFoam import (
    fvc,
    fvm,
    fvScalarMatrix,
    volScalarField,
    volTensorField,
)


def correct(k: Any, epsilon: Any, nut: Any, turb: Any) -> None:
    """Manual kEpsilon correct() using pybFoam — mirrors OF."""
    Cmu = 0.09
    C1 = 1.44
    C2 = 1.92
    sigmaK = 1.0
    sigmaEps = 1.3

    nu_f = volScalarField(turb.nu())
    U = turb.U()
    phi = turb.phi()

    # divU (for compressibility/dilatation terms)
    divU = volScalarField(fvc.div(phi))

    # G = nut * (gradU && devTwoSymm(gradU))
    gradU = volTensorField(fvc.grad(U))
    GbyNu = volScalarField(pyf.doubleInner(gradU, pyf.devTwoSymm(gradU)))
    G = volScalarField(nut * GbyNu)

    # Effective diffusivities
    DepsEff = nut / sigmaEps + nu_f
    DkEff = nut / sigmaK + nu_f

    # Epsilon equation (first)
    # OF: == C1*GbyNu*Cmu*k - SuSp((2/3)*C1*divU, eps) - Sp(C2*eps/k, eps)
    eps_prod = volScalarField((C1 * Cmu) * volScalarField(GbyNu * k))
    eqn_eps = fvScalarMatrix(
        fvm.ddt(epsilon)
        + fvm.div(phi, epsilon)
        - fvm.laplacian(DepsEff, epsilon)
        + fvm.Sp(C2 * epsilon / k, epsilon)
        + fvm.SuSp((2.0 / 3.0 * C1) * divU, epsilon)
        - fvm.Su(1.0 * eps_prod, epsilon)
    )
    eqn_eps.relax()
    eqn_eps.solve()
    pyf.bound(epsilon, pyf.dimensionedScalar("z", pyf.dimViscosity / pyf.dimTime, 1e-10))
    epsilon.correctBoundaryConditions()

    # k equation (second, with updated epsilon)
    # OF: == G - SuSp((2/3)*divU, k) - Sp(eps/k, k)
    eqn_k = fvScalarMatrix(
        fvm.ddt(k)
        + fvm.div(phi, k)
        - fvm.laplacian(DkEff, k)
        + fvm.Sp(epsilon / k, k)
        + fvm.SuSp((2.0 / 3.0) * divU, k)
        - fvm.Su(1.0 * G, k)
    )
    eqn_k.relax()
    eqn_k.solve()
    pyf.bound(k, pyf.dimensionedScalar("z", pyf.dimVelocity * pyf.dimVelocity, 1e-10))
    k.correctBoundaryConditions()

    # Update nut
    nut.assign(Cmu * k * k / epsilon)
    nut.correctBoundaryConditions()
