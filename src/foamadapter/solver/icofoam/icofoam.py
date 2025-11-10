import pybFoam
from pybFoam import (
    Info,
    Time,
    Word,
    adjustPhi,
    constrainHbyA,
    constrainPressure,
    createPhi,
    dictionary,
    fvc,
    fvm,
    fvMesh,
    fvScalarMatrix,
    fvVectorMatrix,
    pisoControl,
    setRefCell,
    solve,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)


def create_fields(mesh):
    p = volScalarField.read_field(mesh, "p")
    U = volVectorField.read_field(mesh, "U")
    phi = createPhi(U)
    nu = volScalarField.read_field(mesh, "nu")  # Assumes viscosity is read like a field

    return p, U, phi, nu


class IcoFoam:
    def __init__(self, argv):
        self._argv = argv

    def run(self):
        argList = pybFoam.argList(self._argv)

        runTime = Time(argList)

        mesh = fvMesh(runTime)

        p, U, phi, nu = create_fields(mesh)

        # Optional: pressure reference

        fvSolution = dictionary.read("system/fvSolution")

        pRefCell, pRefValue = setRefCell(p, fvSolution.subDict("PISO"))

        mesh.setFluxRequired(Word("p"))

        piso = pisoControl(mesh)

        while runTime.loop():
            Info(f"Time = {runTime.timeName()}")

            # Courant number computation assumed handled elsewhere
            # (optionally bind and call selectCourantNo)

            UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) - fvm.laplacian(nu, U))

            if piso.momentumPredictor():
                solve(UEqn + fvc.grad(p))

            while piso.correct():
                rAU = volScalarField(Word("rAU"), 1.0 / UEqn.A())

                HbyA = volVectorField(constrainHbyA(rAU * UEqn.H(), U, p))

                phiHbyA = surfaceScalarField(
                    Word("phiHbyA"), fvc.flux(HbyA) + fvc.interpolate(rAU) * fvc.ddtCorr(U, phi)
                )

                adjustPhi(phiHbyA, U, p)

                constrainPressure(p, U, phiHbyA, rAU)

                while piso.correctNonOrthogonal():
                    pEqn = fvScalarMatrix(fvm.laplacian(rAU, p) - fvc.div(phiHbyA))

                    pEqn.setReference(pRefCell, pRefValue, False)

                    pEqn.solve(p.select(piso.finalInnerIter()))

                    if piso.finalNonOrthogonalIter():
                        phi.assign(phiHbyA - pEqn.flux())

                # Optionally include continuityErrs()

                U.assign(HbyA - rAU * fvc.grad(p))

                U.correctBoundaryConditions()

            runTime.write(True)

            runTime.printExecutionTime()

        Info("End")

