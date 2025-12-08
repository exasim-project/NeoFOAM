# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2023 FoamAdapter authors

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

    def __init__(self, maxDeltaT):
        self.criteria = []
        self.GREAT = 1e30
        self.SMALL = 1e-15
        controlDict = pyf.dictionary.read("system/controlDict")
        self.adjustable = controlDict.get[bool]("adjustTimeStep")
        self.maxCFL = controlDict.get[float]("maxCo")
        self.maxDeltaT = maxDeltaT


    def setDelta(self, runTime, phi, maxRatio=1.2):
        deltaT = runTime.deltaTValue()

        if not self.criteria:
            return

        maxCFLNumber, meanCFLNumber = pyf.computeCFLNumber(phi)
        Info(f"Courant Number mean: {meanCFLNumber}, max: {maxCFLNumber}")
        ratios = [self.maxCFL / maxCFLNumber]

        # limit to 1.2 to avoid too large time steps
        ratios = [min(ratio, maxRatio) for ratio in ratios if ratio > self.SMALL and ratio < self.GREAT]

        # Set most restrictive time step
        finalDeltaT = min(min(deltaT * ratio for ratio in ratios), self.maxDeltaT)
        runTime.setDeltaT(finalDeltaT)
        runTime.increment()


def create_fields(mesh):
    p = volScalarField.read_field(mesh, "p")
    U = volVectorField.read_field(mesh, "U")
    phi = pyf.createPhi(U)

    laminarTransport = singlePhaseTransportModel(U, phi)
    turbulence = incompressibleTurbulenceModel.New(U, phi, laminarTransport)

    return p, U, phi, laminarTransport, turbulence


class PimpleFoam:
    def __init__(self, argv):
        self._argv = argv
        self.pRefCell = None
        self.pRefValue = None

    def momentum_equation(self, pimple, U, p, phi, turbulence) -> fvVectorMatrix:
        """
        Solve the momentum equations using the PIMPLE algorithm.
        """
        UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))

        UEqn.relax()

        if pimple.momentumPredictor():
            pyf.solve(UEqn + fvc.grad(p))

        return UEqn

    def pressure_correction(self, pimple, U, p, phi, UEqn) -> None:
        """
        Correct the solution based on the PIMPLE algorithm.
        """
        rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
        HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p))

        phiHbyA = surfaceScalarField(
            pyf.Word("phiHbyA"), fvc.flux(HbyA) + fvc.interpolate(rAU) * fvc.ddtCorr(U, phi)
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

    def run(self):
        argList = pyf.argList(self._argv)
        runTime = pyf.Time(argList)
        mesh = pyf.fvMesh(runTime)

        p, U, phi, laminarTransport, turbulence = create_fields(mesh)

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
            cfl_number.setDelta(runTime, phi)

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
