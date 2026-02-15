# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Reusable pressure-velocity helper kernels.

Solver ownership and operation orchestration live in model plugins.
This module only contains reusable computational helpers.
"""

from typing import Any

import pybFoam as pyf
from pybFoam import (
    fvc,
    fvm,
    fvScalarMatrix,
    fvVectorMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)


def momentum_helper(
    U: Any,
    phi: Any,
    p: Any,
    turbulence: Any,
    pimple_control: Any,
) -> Any:
    """Assemble and optionally solve momentum predictor for standard formulation."""
    UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))
    UEqn.relax()

    if pimple_control.momentumPredictor():
        pyf.solve(UEqn + fvc.grad(p))

    return UEqn


def continuity_helper(
    U: Any,
    p: Any,
    phi: Any,
    UEqn: Any,
    pimple_control: Any,
    cumulativeContErr: list[float],
    pRefCell: Any,
    pRefValue: Any,
) -> None:
    """Run standard PIMPLE pressure-velocity correction loop."""
    while pimple_control.correct():
        rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
        HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p))

        phiHbyA = surfaceScalarField(
            pyf.Word("phiHbyA"),
            fvc.flux(HbyA) + fvc.interpolate(rAU) * fvc.ddtCorr(U, phi),
        )

        pyf.adjustPhi(phiHbyA, U, p)
        pyf.constrainPressure(p, U, phiHbyA, rAU)

        while pimple_control.correctNonOrthogonal():
            pEqn = fvScalarMatrix(fvm.laplacian(rAU, p) - fvc.div(phiHbyA))
            pEqn.setReference(pRefCell, pRefValue, False)
            pEqn.solve(p.select(pimple_control.finalInnerIter()))

            if pimple_control.finalNonOrthogonalIter():
                phi.assign(phiHbyA - pEqn.flux())

        U.assign(HbyA - rAU * fvc.grad(p))
        U.correctBoundaryConditions()

        sum_local, global_err = pyf.computeContinuityErrors(phi)
        cumulativeContErr[0] += global_err
        pyf.Info(
            f"time step continuity errors : sum local = {sum_local}, "
            f"global = {global_err}, cumulative = {cumulativeContErr[0]}"
        )


def momentum_boussinesq_helper(
    U: Any,
    phi: Any,
    p_rgh: Any,
    rhok: Any,
    ghf: Any,
    turbulence: Any,
    pimple_control: Any,
) -> Any:
    """Assemble and optionally solve momentum predictor for Boussinesq formulation."""
    mesh = U.mesh()
    UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))
    UEqn.relax()

    if pimple_control.momentumPredictor():
        pyf.solve(
            UEqn
            + fvc.reconstruct(
                (-ghf * fvc.snGrad(rhok) - fvc.snGrad(p_rgh)) * mesh.magSf()
            )
        )

    return UEqn


def continuity_boussinesq_helper(
    U: Any,
    p: Any,
    p_rgh: Any,
    phi: Any,
    UEqn: Any,
    rhok: Any,
    gh: Any,
    ghf: Any,
    pimple_control: Any,
    cumulativeContErr: list[float],
    pRefCell: Any,
    pRefValue: Any,
) -> None:
    """Run Boussinesq pressure-velocity correction loop using p_rgh."""
    mesh = U.mesh()

    while pimple_control.correct():
        rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
        rAUf = surfaceScalarField(pyf.Word("rAUf"), fvc.interpolate(rAU))
        HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p_rgh))

        phig = surfaceScalarField(
            pyf.Word("phig"), -rAUf * ghf * fvc.snGrad(rhok) * mesh.magSf()
        )

        phiHbyA = surfaceScalarField(
            pyf.Word("phiHbyA"),
            fvc.flux(HbyA) + rAUf * fvc.ddtCorr(U, phi) + phig,
        )

        pyf.constrainPressure(p_rgh, U, phiHbyA, rAUf)

        while pimple_control.correctNonOrthogonal():
            pEqn = fvScalarMatrix(fvm.laplacian(rAUf, p_rgh) - fvc.div(phiHbyA))
            pEqn.setReference(pRefCell, pRefValue, False)
            pEqn.solve(p_rgh.select(pimple_control.finalInnerIter()))

            if pimple_control.finalNonOrthogonalIter():
                phi.assign(phiHbyA - pEqn.flux())

        U.assign(HbyA + rAU * fvc.reconstruct((phig - pEqn.flux()) / rAUf))
        U.correctBoundaryConditions()
        p.assign(p_rgh + rhok * gh)

        sum_local, global_err = pyf.computeContinuityErrors(phi)
        cumulativeContErr[0] += global_err
        pyf.Info(
            f"time step continuity errors : sum local = {sum_local}, "
            f"global = {global_err}, cumulative = {cumulativeContErr[0]}"
        )
