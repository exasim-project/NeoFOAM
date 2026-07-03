# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from typing import Any

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
from pybFoam.turbulence import incompressibleTurbulenceModel, singlePhaseTransportModel


class CFLNumber:
    def __init__(self, maxDeltaT: float) -> None:
        self.criteria: list[Any] = []
        self.GREAT = 1e30
        self.SMALL = 1e-15
        controlDict = pyf.dictionary.read("system/controlDict")
        self.adjustable = controlDict.get[bool]("adjustTimeStep")
        self.maxCFL = controlDict.get[float]("maxCo")
        self.maxDeltaT = maxDeltaT

    def setDeltaT(
        self, runTime: pyf.Time, phi: pyf.surfaceScalarField, maxRatio: float = 1.2
    ) -> None:
        deltaT = runTime.deltaTValue()

        if not self.criteria:
            return

        maxCFLNumber, meanCFLNumber = pyf.computeCFLNumber(phi)
        Info(f"Courant Number mean: {meanCFLNumber}, max: {maxCFLNumber}")
        ratios = [self.maxCFL / maxCFLNumber]

        # limit to 1.2 to avoid too large time steps
        ratios = [
            min(ratio, maxRatio)
            for ratio in ratios
            if ratio > self.SMALL and ratio < self.GREAT
        ]

        # Set most restrictive time step
        finalDeltaT = min(min(deltaT * ratio for ratio in ratios), self.maxDeltaT)
        runTime.setDeltaT(finalDeltaT)
        runTime.increment()


def create_fields(
    mesh: Any,
) -> tuple[
    volScalarField,
    volVectorField,
    surfaceScalarField,
    singlePhaseTransportModel,
    incompressibleTurbulenceModel,
]:
    p = volScalarField.read_field(mesh, "p")
    U = volVectorField.read_field(mesh, "U")
    phi = pyf.createPhi(U)

    laminarTransport = singlePhaseTransportModel(U, phi)
    turbulence = incompressibleTurbulenceModel.New(U, phi, laminarTransport)

    return p, U, phi, laminarTransport, turbulence


class PimpleFoam:
    def __init__(self, argv: list[str]) -> None:
        self._argv = argv
        self.pRefCell: int = 0
        self.pRefValue: float = 0.0

    def momentum_equation(
        self, pimple: Any, U: Any, p: Any, phi: Any, turbulence: Any
    ) -> fvVectorMatrix:
        """
        Solve the momentum equations using the PIMPLE algorithm.
        """
        UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))

        UEqn.relax()

        if pimple.momentumPredictor():
            pyf.solve(UEqn + fvc.grad(p))

        return UEqn

    def pressure_correction(
        self, pimple: Any, U: Any, p: Any, phi: Any, UEqn: Any
    ) -> None:
        """
        Correct the solution based on the PIMPLE algorithm.
        """
        rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
        HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p))

        phiHbyA = surfaceScalarField(
            pyf.Word("phiHbyA"),
            fvc.flux(HbyA) + fvc.interpolate(rAU) * fvc.ddtCorr(U, phi),
        )

        pyf.adjustPhi(phiHbyA, U, p)
        pyf.constrainPressure(p, U, phiHbyA, rAU)
        while pimple.correctNonOrthogonal():
            pEqn = fvScalarMatrix(fvm.laplacian(rAU, p) - fvc.div(phiHbyA))
            pEqn.setReference(self.pRefCell, self.pRefValue, False)
            pEqn.solve(p.select(pimple.finalInnerIter()))
            if pimple.finalNonOrthogonalIter():
                phi.assign(phiHbyA - pEqn.flux())

        # TODO include continuityErrs()
        U.assign(HbyA - rAU * fvc.grad(p))
        U.correctBoundaryConditions()

    def run(self) -> None:
        argList = pyf.argList(self._argv)
        runTime = pyf.Time(argList)
        mesh = pyf.fvMesh(runTime)

        p, U, phi, laminarTransport, turbulence = create_fields(mesh)
        # Recompute nut from the initial k/epsilon (correctNut) before the first
        # solve, exactly as pimpleFoam does — the 0/nut on disk is typically a
        # placeholder, so without this the first momentum equation runs with
        # nuEff = nu and diverges from the native solver.
        turbulence.validate()

        fvSolution = pyf.dictionary.read("system/fvSolution")
        self.pRefCell, self.pRefValue = pyf.setRefCell(p, fvSolution.subDict("PIMPLE"))
        mesh.setFluxRequired(pyf.Word("p"))
        controlDict = pyf.dictionary.read("system/controlDict")
        maxDeltaT = 1e5
        try:
            maxDeltaT = controlDict.get[float]("maxDeltaT")
        except KeyError:
            pass
        cfl_number = CFLNumber(maxDeltaT)

        pimple = pyf.pimpleControl(mesh)

        while runTime.loop():
            Info(f"Time = {runTime.timeName()}")

            # Compute Courant number
            cfl_number.setDeltaT(runTime, phi)

            while pimple.loop():
                UEqn = self.momentum_equation(pimple, U, p, phi, turbulence)

                while pimple.correct():
                    self.pressure_correction(pimple, U, p, phi, UEqn)

                if pimple.turbCorr():
                    laminarTransport.correct()
                    turbulence.correct()

            runTime.write(True)
            runTime.printExecutionTime()

        Info("End")
