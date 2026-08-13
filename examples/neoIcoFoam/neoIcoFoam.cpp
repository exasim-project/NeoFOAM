// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 nf authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

#include "fvCFD.H"
#include "pisoControl.H"

#include <memory>

using Foam::Info;
using Foam::endl;
using Foam::nl;

// NOTE about namespace usage
// here the namespaces are used deliberately verbose to
// demonstrate where things are implemented
namespace fvc = Foam::fvc;
namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Owns the solver objects and the PISO time loop. Templated independently on the momentum (U)
// and pressure (p) system matrix formats, so all four CSR/ELL combinations are compiled into
// this one binary -- main() dispatches to whichever pair each equation's fvSolution sub-dict
// (solvers.U.matrixFormat / solvers.p.matrixFormat) asks for at startup, with no rebuild needed
// to switch formats. See NeoFOAM::matrixFormat() (fvSolution.hpp) for the dictionary lookup.
template<typename UMatrixType, typename PMatrixType>
void runCase(
    Foam::Time& runTime,
    NeoFOAM::RunTime& rt,
    Foam::pisoControl& piso,
    fvcc::VolumeField<NeoN::Vec3>& U,
    fvcc::VolumeField<NeoN::scalar>& p,
    fvcc::SurfaceField<NeoN::scalar>& nu,
    fvcc::SurfaceField<NeoN::scalar>& phi,
    NeoN::scalar& cumulativeContErr,
    Foam::volScalarField& ofP,
    Foam::label pRefCell,
    Foam::scalar pRefValue,
    NeoFOAM::MeshAdapter& mesh
)
{
    nf::Solver<NeoN::Vec3, NeoN::scalar, NeoN::localIdx, UMatrixType> uSolver(U, rt);
    nf::Solver<NeoN::scalar, NeoN::scalar, NeoN::localIdx, PMatrixType> pSolver(p, rt);

    NeoN::Logging::info("Starting time loop");
    while (runTime.loop())
    {
        // runTime.loop() has already advanced OpenFOAM to the current step. Mirror that into
        // the adapter time before logging: rt.t is set to startTime at construction and only
        // refreshed later in syncRunTimes, so logging it here would report the previous step's
        // time -- making NeoFOAM's time column start at 0 and lag OpenFOAM by one step when
        // comparing per-step timings. syncRunTimes still runs below for the dt adjustment.
        rt.t = runTime.time().value();
        // Logging supports string formatting
        NeoN::Logging::info("Time = {}", rt.t);

        fvcc::rotateOldTimes(U);
        fvcc::rotateOldTimes(phi);

        auto [maxCoNum, meanCoNum] = fvcc::computeCoNum(phi, rt.dt);
        NeoN::Logging::info("Courant Number mean: {} max: {}", meanCoNum, maxCoNum);
        nf::syncRunTimes(runTime, rt, maxCoNum);

        // Momentum predictor
        nf::PDE<NeoN::Vec3, NeoN::scalar, NeoN::localIdx, UMatrixType> UEqn(
            dsl::imp::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(nu, U)
        );

        const auto ddtScheme = UEqn.ddtScheme();

        if (piso.momentumPredictor())
        {
            // NOTE solve on a temporary clone of UEqn; owned LS stores assembly without rhs
            // TODO use a free function here
            uSolver.solve(UEqn, -1.0 * dsl::exp::grad(p));
        }
        else
        {
            // NOTE since computing rAU and HbyA requires an assembled system matrix we
            // explicitly trigger assembly here.
            uSolver.assemble(UEqn);
        }

        // --- PISO loop
        while (piso.correct())
        {
            NeoN::Logging::info("PISO loop");
            auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
            nf::constrainHbyA(U, p, hByA);

            fvcc::SurfaceField<NeoN::scalar> rAU =
                fvcc::SurfaceInterpolation<NeoN::scalar>(
                    rt.exec,
                    rt.nfMesh,
                    NeoN::TokenList({std::string("linear")})
                )
                    .interpolate(crAU);
            rAU.name = "rAUf";

            auto phiHbyA = nf::flux(hByA) + rAU * fvcc::ddtFluxCorr(U, phi, rt.dt, ddtScheme);

            // TODO additionally missing
            // Foam::adjustPhi(phiHbyA, U, p);
            // Update the pressure BCs to ensure flux consistency
            // Foam::constrainPressure(p, U, phiHbyA, rAU);

            // Non-orthogonal pressure corrector loop
            while (piso.correctNonOrthogonal())
            {
                // Pressure corrector
                nf::PDE<NeoN::scalar, NeoN::scalar, NeoN::localIdx, PMatrixType> pEqn(
                    NeoN::dsl::imp::laplacian(rAU, p) - NeoN::dsl::exp::div(phiHbyA)
                );

                // updateFaceVelocity reconstructs phi from this pressure system; keep the
                // non-orthogonal faceFluxCorrection so the reconstruction can add it back.
                // TODO find a more suitable spot
                pEqn.linearSystem().keepFaceFluxCorrection(true);

                if (ofP.needReference() && pRefCell >= 0)
                {
                    pEqn.setReference(pRefCell, pRefValue);
                }

                pSolver.solve(pEqn);
                p.correctBoundaryConditions();

                if (piso.finalNonOrthogonalIter())
                {
                    nf::updateFaceVelocity(phiHbyA, pEqn, phi);
                }
            }
            nf::reportContinuityError(phi, rt, cumulativeContErr);

            nf::updateVelocity(hByA, crAU, p, U);
            U.correctBoundaryConditions();
        }

        runTime.write();
        if (runTime.outputTime())
        {
            NeoN::Logging::info("Writing p");
            write(p, mesh);
            NeoN::Logging::info("Writing U");
            write(U, mesh);
        }

        runTime.printExecutionTime(Info);
    }
}

// Dispatches to the runCase<UMatrixType, PMatrixType> instantiation matching the formats read
// from fvSolution. Only four combinations exist (CSR/ELL for each of U and p), so a plain
// switch is clearer here than a factory map -- see runCase's doc comment.
void dispatchMatrixFormats(
    NeoFOAM::MatrixFormat uFormat,
    NeoFOAM::MatrixFormat pFormat,
    Foam::Time& runTime,
    NeoFOAM::RunTime& rt,
    Foam::pisoControl& piso,
    fvcc::VolumeField<NeoN::Vec3>& U,
    fvcc::VolumeField<NeoN::scalar>& p,
    fvcc::SurfaceField<NeoN::scalar>& nu,
    fvcc::SurfaceField<NeoN::scalar>& phi,
    NeoN::scalar& cumulativeContErr,
    Foam::volScalarField& ofP,
    Foam::label pRefCell,
    Foam::scalar pRefValue,
    NeoFOAM::MeshAdapter& mesh
)
{
    using CsrMatrix = NeoN::la::CSRMatrix<NeoN::scalar, NeoN::localIdx>;
    using EllMatrix = NeoN::la::ELLMatrix<NeoN::scalar, NeoN::localIdx>;

    if (uFormat == NeoFOAM::MatrixFormat::CSR && pFormat == NeoFOAM::MatrixFormat::CSR)
    {
        runCase<CsrMatrix, CsrMatrix>(
            runTime, rt, piso, U, p, nu, phi, cumulativeContErr, ofP, pRefCell, pRefValue, mesh
        );
    }
    else if (uFormat == NeoFOAM::MatrixFormat::CSR && pFormat == NeoFOAM::MatrixFormat::ELL)
    {
        runCase<CsrMatrix, EllMatrix>(
            runTime, rt, piso, U, p, nu, phi, cumulativeContErr, ofP, pRefCell, pRefValue, mesh
        );
    }
    else if (uFormat == NeoFOAM::MatrixFormat::ELL && pFormat == NeoFOAM::MatrixFormat::CSR)
    {
        runCase<EllMatrix, CsrMatrix>(
            runTime, rt, piso, U, p, nu, phi, cumulativeContErr, ofP, pRefCell, pRefValue, mesh
        );
    }
    else
    {
        runCase<EllMatrix, EllMatrix>(
            runTime, rt, piso, U, p, nu, phi, cumulativeContErr, ofP, pRefCell, pRefValue, mesh
        );
    }
}

int main(int argc, char* argv[])
{
    Foam::argList::addOption("executor", "word", "NeoN executor type (Serial/CPU/GPU/default)");
#include "addCheckCaseOptions.H"
#include "setRootCase.H"
    NeoN::initialize(argc, argv);
    {
// createTime.H is included INSIDE the NeoN init/finalize bracket so Foam::Time --
// which owns the controlDict function objects (e.g. neoForceCoeffs) and therefore any
// NeoN/Kokkos memory they hold -- is destroyed before NeoN::finalize() calls
// Kokkos::finalize(). Otherwise ~Time() runs after finalize and aborts with
// "Kokkos allocation ... deallocated after Kokkos::finalize was called".
#include "createTime.H"
        auto rt = nf::createAdapterRunTime(runTime, args);
        auto& mesh = rt.mesh;

        Foam::pisoControl piso(mesh);

#include "createFields.H"


        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(rt.db, "VectorCollection");

        auto& p = nf::constructAndRegister(vectorCollection, rt, ofP, false);
        auto& U = nf::constructAndRegister(vectorCollection, rt, ofU, false);

        auto nuBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh);
        fvcc::SurfaceField<NeoN::scalar> nu(rt.exec, "nu", rt.nfMesh, nuBCs);
        NeoN::fill(nu.internalVector(), viscosity.value());
        NeoN::fill(nu.boundaryData().value(), viscosity.value());

        NeoN::Logging::info("Creating phi");
        auto& phi = nf::constructAndRegister(vectorCollection, rt, ofPhi, false);
        NeoN::scalar cumulativeContErr = 0.0;

        // matrixFormat is read from each equation's own sub-dict (solvers.U.matrixFormat,
        // solvers.p.matrixFormat) independently -- see NeoFOAM::matrixFormat's doc comment.
        // Safe to read after mapFvSolution: that mapping never touches this key. Default is
        // ELL, not the library-wide CSR default: neoIcoFoam hardcoded ELL for both equations
        // before this option existed, so cases without the key keep that prior behavior.
        const auto uFormat = nf::matrixFormat(solverDict.subDict("U"), nf::MatrixFormat::ELL);
        const auto pFormat = nf::matrixFormat(solverDict.subDict("p"), nf::MatrixFormat::ELL);
        NeoN::Logging::info(
            "Matrix format: U={} p={}",
            uFormat == nf::MatrixFormat::ELL ? "ELL" : "CSR",
            pFormat == nf::MatrixFormat::ELL ? "ELL" : "CSR"
        );
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        dispatchMatrixFormats(
            uFormat, pFormat, runTime, rt, piso, U, p, nu, phi, cumulativeContErr, ofP, pRefCell,
            pRefValue, mesh
        );
    }
    NeoN::finalize();

    return 0;
}

// ************************************************************************* //
