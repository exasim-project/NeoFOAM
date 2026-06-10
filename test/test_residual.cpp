// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include <cstddef>
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

// Verifies that NeoN's L1-scaled residual stopping criterion reports the same
// (normFactor-scaled, L1) initial residual that OpenFOAM's solverPerformance
// reports for the identical assembled linear system.
TEST_CASE("L1 scaled residual matches OpenFOAM")
{
    NeoN::mpi::Environment mpiEnviron;
    Foam::Time& runTime = *timePtr;

    auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto rt = nf::createAdapterRunTime(runTime, exec);
    fvcc::VectorCollection& fieldCol = fvcc::VectorCollection::instance(rt.db, "VectorCollection");

    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    SECTION("laplacian initial residual " + execName)
    {
        auto ofT = NeoFOAM::randomScalarField(runTime, mesh, "T");
        ofT.correctBoundaryConditions();
        auto ofGamma =
            NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, Foam::dimless, "Gamma");

        Foam::fvScalarMatrix matrix(Foam::fvm::laplacian(ofGamma, ofT));

        // Build the matching NeoN system from the SAME initial field, before the
        // OpenFOAM solve below evaluates its residual.
        auto [nfT, nfGamma] = NeoFOAM::constFromMany(exec, rt.nfMesh, ofT, ofGamma);
        auto nfPDE = NeoFOAM::PDESolver<NeoN::scalar>(
            NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::laplacian(nfGamma, nfT)),
            nfT,
            rt
        );
        nfPDE.assemble();

        // OpenFOAM ground truth: solverPerformance::initialResidual() is the scaled L1
        // residual evaluated at the starting field (iteration 0), regardless of how many
        // iterations the solve then performs. ofT was already copied into nfT above, so
        // the solve below mutating ofT does not affect the comparison.
        Foam::dictionary solverDict;
        solverDict.add("solver", Foam::word("PCG"));
        solverDict.add("preconditioner", Foam::word("DIC"));
        solverDict.add("tolerance", Foam::scalar(1e-20));
        solverDict.add("relTol", Foam::scalar(0.0));
        solverDict.add("maxIter", Foam::label(500));
        const Foam::scalar ofInitResidual = matrix.solve(solverDict).initialResidual();

        // NeoN: solve with the L1-scaled residual stopping criterion enabled and the
        // iteration cap at 0, so the reported initial residual is the scaled L1 residual
        // of the assembled system at the starting field.
        NeoN::Dictionary solverConfig {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"l1ScaledResidual", true},
             {"preconditioner",
              NeoN::Dictionary {{{"type", "preconditioner::Jacobi"}, {"max_block_size", 1}}}},
             {"criteria", NeoN::Dictionary {{{"iteration", 0}, {"absolute_residual_norm", 1e-9}}}}}
        };
        auto solver = NeoN::la::Solver(exec, solverConfig);
        auto stats = solver.solve(nfPDE.linearSystem(), nfT.internalVector());

        REQUIRE(stats.entries.size() >= 1);
        const NeoN::scalar nfInitResidual = stats.entries[0].initResNorm;

        REQUIRE(nfInitResidual == Catch::Approx(ofInitResidual).epsilon(1e-6).margin(1e-12));

        // Dictionary form: a boolean read from a dictionary file (fvSolution) arrives as a
        // word/string, not a bool. Enabling via "l1ScaledResidual" as a string must work
        // identically, otherwise the feature is unreachable from case dictionaries.
        NeoN::Dictionary solverConfigStr {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"l1ScaledResidual", std::string {"true"}},
             {"preconditioner",
              NeoN::Dictionary {{{"type", "preconditioner::Jacobi"}, {"max_block_size", 1}}}},
             {"criteria", NeoN::Dictionary {{{"iteration", 0}, {"absolute_residual_norm", 1e-9}}}}}
        };
        auto solverStr = NeoN::la::Solver(exec, solverConfigStr);
        auto statsStr = solverStr.solve(nfPDE.linearSystem(), nfT.internalVector());
        REQUIRE(statsStr.entries.size() >= 1);
        REQUIRE(
            statsStr.entries[0].initResNorm
            == Catch::Approx(ofInitResidual).epsilon(1e-6).margin(1e-12)
        );
    }

    SECTION("L1 stopping criterion converges below tolerance " + execName)
    {
        auto ofT = NeoFOAM::randomScalarField(runTime, mesh, "T");
        ofT.correctBoundaryConditions();
        auto ofGamma =
            NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, Foam::dimless, "Gamma");

        auto [nfT, nfGamma] = NeoFOAM::constFromMany(exec, rt.nfMesh, ofT, ofGamma);
        auto nfPDE = NeoFOAM::PDESolver<NeoN::scalar>(
            NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::laplacian(nfGamma, nfT)),
            nfT,
            rt
        );
        nfPDE.assemble();

        const NeoN::scalar absTol = 1e-8;
        NeoN::Dictionary solverConfig {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"l1ScaledResidual", true},
             {"preconditioner",
              NeoN::Dictionary {{{"type", "preconditioner::Jacobi"}, {"max_block_size", 1}}}},
             {"criteria",
              NeoN::Dictionary {{{"iteration", 1000}, {"absolute_residual_norm", absTol}}}}}
        };
        auto solver = NeoN::la::Solver(exec, solverConfig);
        auto stats = solver.solve(nfPDE.linearSystem(), nfT.internalVector());

        REQUIRE(stats.entries.size() >= 1);
        // the criterion must drive the scaled residual below the absolute tolerance
        REQUIRE(stats.entries[0].finalResNorm < absTol);
        REQUIRE(stats.entries[0].numIter > 0);
    }
}
