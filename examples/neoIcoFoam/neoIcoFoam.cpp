// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 nf authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"
#include "NeoFOAM/auxiliary/procFaceCheck.hpp"
#include "NeoFOAM/auxiliary/procFaceDump.hpp"
#include "NeoFOAM/auxiliary/fullDump.hpp"
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

        // FULL DUMP: one-shot geometry + per-rank meta. Step 0, written once.
        //   Gated on NEOFOAM_FULL_DUMP=1 (no-op when unset). Captures sf,
        //   magSf, cf, faceCells, deltaCoeffs, weights AND a per-rank
        //   proc_meta.txt with the range-partition row offset that the
        //   standalone Ginkgo CG reproducer (tools/ginkgo-standalone-cg/)
        //   reads back to rebuild a distributed gko matrix from a dump tree.
        nf::dumpGeometry(rt.nfMesh);
        nf::dumpProcMeta(rt.nfMesh);

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
            // FULL DUMP: U + phi internal + all faces at the entry to the time step.
            nf::dumpInternal(U,    "U",   "after_rotateOldTimes", stepIdx, 0, 0);
            nf::dumpInternal(phi,  "phi", "after_rotateOldTimes", stepIdx, 0, 0);
            nf::dumpAllFaces(phi,  "phi", "after_rotateOldTimes", stepIdx, 0, 0);

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

            // DUMP: the standalone explicit grad(p) source as an explicit
            //   Vector<Vec3> field. At step 1 with p=0 internally, grad(p)
            //   should be 0 everywhere. If it is non-zero on GPU at proc-
            //   adjacent cells (and zero on CPU), the gradient operator is
            //   reading uninitialised p.boundaryData() at the proc tail OR
            //   using wrong proc-face indexing. Both classic NeoN bugs.
            {
                auto gradPExpr = NeoN::dsl::Expression<NeoN::Vec3>(-1.0 * dsl::exp::grad(p));
                gradPExpr.read(rt.fvSchemesDict);
                auto gradPSource = gradPExpr.explicitOperation(mesh.nCells());
                nf::dumpVector(
                    gradPSource, "grad_p_source", "before_UEqn_solve", stepIdx, 0, 0
                );
            }

            // DUMP: the FULL LinearSystem solve(rhs) actually hands to Ginkgo
            //   (implicit ls_ + explicit grad(p) folded into rhs). If this
            //   rhs differs CPU/GPU while the implicit-only rhs above matched,
            //   the gradient operator is producing different values on GPU.
            //   This is the dump that captures whatever solve(rhs) sees right
            //   before passing it to the Krylov solver.
            {
                auto rhsLS = UEqn.assemble(-1.0 * dsl::exp::grad(p));
                nf::dumpFullLinearSystem(
                    rhsLS,
                    U.internalVector(),
                    "U",
                    "after_UEqn_assemble_with_grad",
                    stepIdx,
                    0,
                    0
                );

                // FULL DUMP: distributed-form CSR + b + x0 + per-rank partition
                //   for `tools/ginkgo-standalone-cg/` replay. This is THE input
                //   Ginkgo's distributed CG receives — diff CPU vs GPU here to
                //   localise: matrix assembly bug (A,b differ) vs Ginkgo bug
                //   (A,b match but solver produces different x).
                nf::dumpDistLinearSystem(
                    rhsLS,
                    U.internalVector(),
                    "U",
                    "before_UEqn_solve",
                    stepIdx,
                    0,
                    0
                );
            }

            if (piso.momentumPredictor())
            {
                // NOTE solve on a temporary clone of UEqn
                // TODO use a free function here
                UEqn.solve(-1.0 * dsl::exp::grad(p));

                // DUMP: U.internalVector() right after Ginkgo solve.
                //   This IS Ginkgo's distributed solver output (the iterate
                //   it wrote back into psi.internalVector()). Compare CPU vs
                //   GPU here to see if the solver itself diverged before
                //   any subsequent step touches the field.
                nf::dumpVector(
                    U.internalVector(), "U_internal", "after_UEqn_solve", stepIdx, 0, 0
                );
                nf::dumpInternal(U, "U", "after_UEqn_solve", stepIdx, 0, 0);  // FULL DUMP

                nf::dumpProcOwnerInternal(U, "U", "after_momentumSolve_preCorrectBC", stepIdx, 0, 0);
                U.correctBoundaryConditions();

                // DUMP: full U state after correctBoundaryConditions().
                //   internal + boundary. Compare CPU vs GPU. If internal
                //   matches but boundary differs => bug is in the proc-tail
                //   exchange path (correctBoundaryConditions). If both
                //   differ identically to "after_UEqn_solve" => correctBC
                //   is just propagating the prior divergence.
                nf::dumpVector(
                    U.internalVector(), "U_internal", "after_U_correctBC", stepIdx, 0, 0
                );
                nf::dumpVector(
                    U.boundaryData().value(), "U_boundary", "after_U_correctBC", stepIdx, 0, 0
                );
                nf::dumpInternal(U, "U", "after_U_correctBC", stepIdx, 0, 0);  // FULL DUMP

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
                nf::dumpInternal(crAU, "rAU",  "after_computeRAUandHByA", stepIdx, pisoIter, 0);   // FULL
                nf::dumpInternal(hByA, "hByA", "after_computeRAUandHByA", stepIdx, pisoIter, 0);   // FULL

                nnfvcc::SurfaceField<NeoN::scalar> rAU =
                    fvcc::SurfaceInterpolation<NeoN::scalar>(
                        rt.exec,
                        rt.nfMesh,
                        NeoN::TokenList({std::string("linear")})
                    )
                        .interpolate(crAU);
                rAU.name = "rAUf";
                nf::dumpProcFaces(rAU, "rAUf", "after_rAU_interpolate", stepIdx, pisoIter, 0);  // DUMP
                nf::dumpAllFaces(rAU, "rAUf", "after_rAU_interpolate", stepIdx, pisoIter, 0);  // FULL

                auto phiHbyA = nf::flux(hByA) + rAU * fvcc::ddtFluxCorr(U, phi, rt.dt, ddtScheme);
                nf::dumpProcFaces(phiHbyA, "phiHbyA", "after_phiHbyA_construct", stepIdx, pisoIter, 0);  // DUMP — KEY for suspect #1
                nf::dumpAllFaces(phiHbyA, "phiHbyA", "after_phiHbyA_construct", stepIdx, pisoIter, 0);  // FULL

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

                    // DUMP: pre-assemble pEqn so we can capture EVERY input
                    //   handed to Ginkgo for the pressure solve, identical
                    //   to what we do for UEqn above. solve() will re-assemble
                    //   internally; coefficients are deterministic so the
                    //   dump matches what Ginkgo sees.
                    pEqn.assemble();
                    nf::dumpFullLinearSystem(
                        pEqn.linearSystem(),
                        p.internalVector(),
                        "p",
                        "after_pEqn_assemble",
                        stepIdx,
                        pisoIter,
                        nonOrthIter
                    );

                    // FULL DUMP: distributed-form CSR for pEqn — symmetric
                    //   counterpart to the UEqn dump above. Same purpose:
                    //   standalone Ginkgo CG reproducer reads this back to
                    //   replay the exact solve.
                    nf::dumpDistLinearSystem(
                        pEqn.linearSystem(),
                        p.internalVector(),
                        "p",
                        "before_pEqn_solve",
                        stepIdx,
                        pisoIter,
                        nonOrthIter
                    );

                    auto stats = pEqn.solve();

                    // DUMP: p.internalVector() right after Ginkgo solve.
                    //   Pressure solver output before any BC correction.
                    nf::dumpVector(
                        p.internalVector(),
                        "p_internal",
                        "after_pEqn_solve",
                        stepIdx,
                        pisoIter,
                        nonOrthIter
                    );
                    nf::dumpInternal(
                        p, "p", "after_pEqn_solve", stepIdx, pisoIter, nonOrthIter
                    );  // FULL DUMP

                    p.correctBoundaryConditions();

                    // DUMP: full p state after correctBoundaryConditions().
                    nf::dumpVector(
                        p.internalVector(),
                        "p_internal",
                        "after_p_correctBC",
                        stepIdx,
                        pisoIter,
                        nonOrthIter
                    );
                    nf::dumpVector(
                        p.boundaryData().value(),
                        "p_boundary",
                        "after_p_correctBC",
                        stepIdx,
                        pisoIter,
                        nonOrthIter
                    );
                    nf::dumpInternal(
                        p, "p", "after_p_correctBC", stepIdx, pisoIter, nonOrthIter
                    );  // FULL DUMP

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
                        nf::dumpAllFaces(phi, "phi", "after_updateFaceVelocity", stepIdx, pisoIter, nonOrthIter);  // FULL
                    }
                }
                // [PISO-03] Continuity error
                nf::reportContinuityError(phi, rt, cumulativeContErr);

                nf::updateVelocity(hByA, crAU, p, U);
                U.correctBoundaryConditions();
                nf::checkProcFaceConsistency(U, "U after updateVelocity");
                nf::dumpProcFaces(U, "U", "after_updateVelocity", stepIdx, pisoIter, 0);  // DUMP
                nf::dumpInternal(U, "U",   "after_updateVelocity", stepIdx, pisoIter, 0);  // FULL
                nf::dumpAllFaces(phi, "phi", "after_updateVelocity", stepIdx, pisoIter, 0); // FULL
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
