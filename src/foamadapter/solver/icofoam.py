# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors
import pybFoam as pyf
from pybFoam import (
    Info,
    fvc,
    fvm,
    fvScalarMatrix,
    fvVectorMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)


def create_fields(
    mesh: pyf.fvMesh,
) -> tuple[volScalarField, volVectorField, surfaceScalarField, volScalarField]:
    p = volScalarField.read_field(mesh, "p")
    U = volVectorField.read_field(mesh, "U")
    phi = pyf.createPhi(U)
    nu = volScalarField.read_field(mesh, "nu")  # Assumes viscosity is read like a field

    return p, U, phi, nu


class IcoFoam:
    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def run(self) -> None:
        argList = pyf.argList(self._argv)

        runTime = pyf.Time(argList)

        mesh = pyf.fvMesh(runTime)

        p, U, phi, nu = create_fields(mesh)

        fvSolution = pyf.dictionary.read("system/fvSolution")

        pRefCell, pRefValue = pyf.setRefCell(p, fvSolution.subDict("PISO"))

        mesh.setFluxRequired(pyf.Word("p"))

        piso = pyf.pisoControl(mesh)

        while runTime.loop():
            Info(f"Time = {runTime.timeName()}")

            UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) - fvm.laplacian(nu, U))

            if piso.momentumPredictor():
                pyf.solve(UEqn + fvc.grad(p))

            while piso.correct():
                rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())

                HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p))

                phiHbyA = surfaceScalarField(
                    pyf.Word("phiHbyA"),
                    fvc.flux(HbyA) + fvc.interpolate(rAU) * fvc.ddtCorr(U, phi),
                )

                pyf.adjustPhi(phiHbyA, U, p)

                pyf.constrainPressure(p, U, phiHbyA, rAU)

                while piso.correctNonOrthogonal():
                    pEqn = fvScalarMatrix(fvm.laplacian(rAU, p) - fvc.div(phiHbyA))

                    pEqn.setReference(pRefCell, pRefValue, False)

                    pEqn.solve(p.select(piso.finalInnerIter()))

                    if piso.finalNonOrthogonalIter():
                        phi.assign(phiHbyA - pEqn.flux())

                # TODO include continuityErrs()

                U.assign(HbyA - rAU * fvc.grad(p))

                U.correctBoundaryConditions()

            runTime.write(True)

            runTime.printExecutionTime()

        Info("End")
