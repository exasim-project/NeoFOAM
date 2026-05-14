// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 nf authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"
#include "NeoFOAM/auxiliary/procFaceCheck.hpp"
#include "NeoFOAM/auxiliary/procFaceDump.hpp"
#include "NeoFOAM/auxiliary/continuityError.hpp"

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

int main(int argc, char* argv[])
{
#include "addCheckCaseOptions.H"
#include "setRootCase.H"
#include "createTime.H"
    NeoN::initialize(argc, argv);
    {
        auto rt = nf::createAdapterRunTime(runTime);
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

        auto commPattern = createCommunicationPattern(rt);

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        NeoN::scalar cumulativeContErr = 0.0;

        NeoN::Logging::info("Starting time loop");
        int stepIdx = 0;  // DUMP: time-step counter for proc-face dumps
        while (runTime.loop())
        {
            // Logging supports string formatting
            NeoN::Logging::info("Time = {}", rt.t);

            ++stepIdx;             // DUMP
            int pisoIter = 0;      // DUMP: reset per time step

            fvcc::rotateOldTimes(U);
            fvcc::rotateOldTimes(phi);
            nf::checkProcFaceConsistency(U, "U after rotateOldTimes");
            nf::checkProcFaceConsistency(
                phi, "phi after rotateOldTimes", 1e-12, nf::SignConvention::FlipExpected
            );
            nf::dumpProcFaces(U,   "U",   "after_rotateOldTimes", stepIdx, 0, 0);  // DUMP
            nf::dumpProcFaces(phi, "phi", "after_rotateOldTimes", stepIdx, 0, 0);  // DUMP

            auto [maxCoNum, meanCoNum] = fvcc::computeCoNum(phi, rt.dt);
            NeoN::Logging::info("Courant Number mean: {} max: {}", meanCoNum, maxCoNum);
            nf::syncRunTimes(runTime, rt, maxCoNum);

            // Momentum predictor
            nf::PDESolver<NeoN::Vec3> UEqn(
                dsl::imp::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(nu, U),
                U,
                rt
            );

            const auto ddtScheme = UEqn.ddtScheme();

            // DUMP: explicitly pre-assemble so we can capture EVERY input
            //   handed to Ginkgo (local CSR values + sparsity, nonLocal COO
            //   values + sparsity, RHS, initial iterate x, and the host-side
            //   commPattern). Diff these CPU vs GPU on the same rank to
            //   identify whether the bug is in our data prep or downstream
            //   in Ginkgo's distributed CUDA solve.
            UEqn.assemble();
            nf::dumpFullLinearSystem(
                UEqn.linearSystem(),
                U.internalVector(),
                "U",
                "after_UEqn_assemble",
                stepIdx,
                0,
                0
            );

            if (piso.momentumPredictor())
            {
                // NOTE solve on a temporary clone of UEqn
                // TODO use a free function here
                UEqn.solve(-1.0 * dsl::exp::grad(p));
                // DUMP: pre-correctBC owner-internal at proc-tail faces.
                //   This is the source value correctBoundaryConditions() will
                //   send to the neighbour rank. Compared against the post-
                //   correctBC boundaryData dump below, this discriminates a
                //   solver bug (pre already wrong) from an exchange bug
                //   (pre matches across CPU/GPU but post diverges).
                nf::dumpProcOwnerInternal(U, "U", "after_momentumSolve_preCorrectBC", stepIdx, 0, 0);
                U.correctBoundaryConditions();
                nf::checkProcFaceConsistency(U, "U after momentumPredictor solve");
                nf::dumpProcFaces(U, "U", "after_momentumPredictor", stepIdx, 0, 0);  // DUMP
            }

            // --- PISO loop
            while (piso.correct())
            {
                NeoN::Logging::info("PISO loop");
                ++pisoIter;            // DUMP
                int nonOrthIter = 0;   // DUMP: reset per PISO outer iter
                auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
                nf::constrainHbyA(U, p, hByA);
                nf::dumpProcFaces(crAU, "rAU",  "after_computeRAUandHByA", stepIdx, pisoIter, 0);  // DUMP
                nf::dumpProcFaces(hByA, "hByA", "after_computeRAUandHByA", stepIdx, pisoIter, 0);  // DUMP

                nnfvcc::SurfaceField<NeoN::scalar> rAU =
                    fvcc::SurfaceInterpolation<NeoN::scalar>(
                        rt.exec,
                        rt.nfMesh,
                        NeoN::TokenList({std::string("linear")})
                    )
                        .interpolate(crAU);
                rAU.name = "rAUf";
                nf::dumpProcFaces(rAU, "rAUf", "after_rAU_interpolate", stepIdx, pisoIter, 0);  // DUMP

                auto phiHbyA = nf::flux(hByA) + rAU * fvcc::ddtFluxCorr(U, phi, rt.dt, ddtScheme);
                nf::dumpProcFaces(phiHbyA, "phiHbyA", "after_phiHbyA_construct", stepIdx, pisoIter, 0);  // DUMP — KEY for suspect #1

                // TODO additionally missing
                // Foam::adjustPhi(phiHbyA, U, p);
                // Update the pressure BCs to ensure flux consistency
                // Foam::constrainPressure(p, U, phiHbyA, rAU);

                // Non-orthogonal pressure corrector loop
                while (piso.correctNonOrthogonal())
                {
                    ++nonOrthIter;  // DUMP
                    // Pressure corrector
                    nf::PDESolver<NeoN::scalar> pEqn(
                        NeoN::dsl::imp::laplacian(rAU, p) - NeoN::dsl::exp::div(phiHbyA),
                        p,
                        rt
                    );

                    if (ofP.needReference() && pRefCell >= 0)
                    {
                        pEqn.setReference(pRefCell, pRefValue);
                    }

                    auto stats = pEqn.solve();
                    p.correctBoundaryConditions();
                    nf::checkProcFaceConsistency(p, "p after correctBC");
                    nf::dumpProcFaces(p, "p", "after_pSolve", stepIdx, pisoIter, nonOrthIter);  // DUMP

                    if (piso.finalNonOrthogonalIter())
                    {
                        nf::updateFaceVelocity(phiHbyA, pEqn, phi);
                        nf::checkProcFaceConsistency(
                            phi,
                            "phi after updateFaceVelocity",
                            1e-12,
                            nf::SignConvention::FlipExpected
                        );
                        nf::dumpProcFaces(phi, "phi", "after_updateFaceVelocity", stepIdx, pisoIter, nonOrthIter);  // DUMP
                    }
                }
                // [PISO-03] Continuity error
                nf::reportContinuityError(phi, rt, cumulativeContErr);

                nf::updateVelocity(hByA, crAU, p, U);
                U.correctBoundaryConditions();
                nf::checkProcFaceConsistency(U, "U after updateVelocity");
                nf::dumpProcFaces(U, "U", "after_updateVelocity", stepIdx, pisoIter, 0);  // DUMP
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
    NeoN::finalize();

    return 0;
}

// ************************************************************************* //
