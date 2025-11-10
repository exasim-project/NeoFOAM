
import pybFoam
from pybFoam import (
    Info,
    Time,
    Word,
    adjustPhi,
    computeCFLNumber,
    constrainHbyA,
    constrainPressure,
    createPhi,
    dictionary,
    fvc,
    fvm,
    fvMesh,
    fvScalarMatrix,
    fvVectorMatrix,
    pimpleControl,
    setRefCell,
    solve,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)
from pybFoam.turbulence import incompressibleTurbulenceModel, singlePhaseTransportModel


class CFLNumber:

    def __init__(self):
        self.criteria = []
        self.GREAT = 1e30
        self.SMALL = 1e-15
        self.maxDeltaT = 1e-3  # Default maximum deltaT

    def setDelta(self, runTime, phi):
        deltaT = runTime.deltaTValue()

        if not self.criteria:
            return

        max_cfl_number, mean_cfl_number = computeCFLNumber(phi)
        Info(f"Courant Number mean: {mean_cfl_number}, max: {max_cfl_number}")
        ratios = [self.max_cfl_number / max_cfl_number]

        # limit to 1.2 to avoid too large time steps
        ratios = [min(ratio, 1.2) for ratio in ratios if ratio > self.SMALL and ratio < self.GREAT]

        # Set most restrictive time step
        finalDeltaT = min(min(deltaT * ratio for ratio in ratios), self.maxDeltaT)
        runTime.setDeltaT(finalDeltaT)
        Info(f"deltaT = {runTime.deltaTValue()}")
        runTime.increment()


def create_fields(mesh):
    p = volScalarField.read_field(mesh, "p")
    U = volVectorField.read_field(mesh, "U")
    phi = createPhi(U)

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
            solve(UEqn + fvc.grad(p))

        return UEqn

    def pressure_correction(self, pimple, U, p, phi, UEqn) -> None:
        """
        Correct the solution based on the PIMPLE algorithm.
        """
        rAU = volScalarField(Word("rAU"), 1.0 / UEqn.A())
        HbyA = volVectorField(constrainHbyA(rAU * UEqn.H(), U, p))

        phiHbyA = surfaceScalarField(
            Word("phiHbyA"), fvc.flux(HbyA) + fvc.interpolate(rAU) * fvc.ddtCorr(U, phi)
        )

        adjustPhi(phiHbyA, U, p)
        constrainPressure(p, U, phiHbyA, rAU)
        while pimple.correctNonOrthogonal():
            pEqn = fvScalarMatrix(fvm.laplacian(rAU, p) - fvc.div(phiHbyA))
            pEqn.setReference(self.pRefCell, self.pRefValue, False)
            pEqn.solve(p.select(pimple.finalInnerIter()))
            if pimple.finalNonOrthogonalIter():
                phi.assign(phiHbyA - pEqn.flux())

        # Optionally include continuityErrs()
        U.assign(HbyA - rAU * fvc.grad(p))
        U.correctBoundaryConditions()

    def run(self):
        argList = pybFoam.argList(self._argv)
        runTime = Time(argList)
        mesh = fvMesh(runTime)

        p, U, phi, laminarTransport, turbulence = create_fields(mesh)

        fvSolution = dictionary.read("system/fvSolution")
        self.pRefCell, self.pRefValue = setRefCell(p, fvSolution.subDict("PIMPLE"))
        mesh.setFluxRequired(Word("p"))
        cfl_number = CFLNumber()

        pimple = pimpleControl(mesh)

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
