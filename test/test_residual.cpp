// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include <cstddef>
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr; // A single time object

// Verifies that NeoN's L1-scaled residual stopping criterion reports the same
// (normFactor-scaled, L1) initial residual that OpenFOAM's solverPerformance
// reports for the identical assembled linear system, for both a scalar laplacian
// and the full vector momentum equation (ddt + div - laplacian).
TEST_CASE("L1 scaled residual matches OpenFOAM")
{
    Foam::Time& runTime = *timePtr;

    auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);
    fvcc::VectorCollection& fieldCol = fvcc::VectorCollection::instance(rt.db, "VectorCollection");

    SECTION("laplacian initial residual " + execName)
    {
        auto ofP = nf::randomScalarField(runTime, mesh, "p");
        ofP.correctBoundaryConditions();
        auto ofGamma = nf::randDimField<Foam::surfaceScalarField>(mesh, Foam::dimless, "Gamma");

        Foam::fvScalarMatrix matrix(Foam::fvm::laplacian(ofGamma, ofP));

        // Build the matching NeoN system from the SAME initial field, before the
        // OpenFOAM solve below evaluates its residual.
        auto [nfP, nfGamma] = nf::constFromMany(exec, rt.nfMesh, ofP, ofGamma);
        auto nfPDE = nf::PDESolver<NeoN::scalar>(
            NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::laplacian(nfGamma, nfP)),
            nfP,
            rt
        );
        nfPDE.assemble();

        // OpenFOAM ground truth: solverPerformance::initialResidual() is the scaled L1
        // residual evaluated at the starting field (iteration 0), regardless of how many
        // iterations the solve then performs. ofP was already copied into nfP above, so
        // the solve below mutating ofP does not affect the comparison.
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
        auto stats = solver.solve(nfPDE.linearSystem(), nfP.internalVector());

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
        auto statsStr = solverStr.solve(nfPDE.linearSystem(), nfP.internalVector());
        REQUIRE(statsStr.entries.size() >= 1);
        REQUIRE(
            statsStr.entries[0].initResNorm
            == Catch::Approx(ofInitResidual).epsilon(1e-6).margin(1e-12)
        );
    }

    SECTION("L1 stopping criterion converges below tolerance " + execName)
    {
        auto ofP = nf::randomScalarField(runTime, mesh, "p");
        ofP.correctBoundaryConditions();
        auto ofGamma = nf::randDimField<Foam::surfaceScalarField>(mesh, Foam::dimless, "Gamma");

        auto [nfP, nfGamma] = nf::constFromMany(exec, rt.nfMesh, ofP, ofGamma);
        auto nfPDE = nf::PDESolver<NeoN::scalar>(
            NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::laplacian(nfGamma, nfP)),
            nfP,
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
        auto stats = solver.solve(nfPDE.linearSystem(), nfP.internalVector());

        REQUIRE(stats.entries.size() >= 1);
        // the criterion must drive the scaled residual below the absolute tolerance
        REQUIRE(stats.entries[0].finalResNorm < absTol);
        REQUIRE(stats.entries[0].numIter > 0);
    }

    SECTION("momentum L1 residual matches OpenFOAM " + execName)
    {
        auto ofU = nf::randomVectorField(runTime, mesh, "U");
        ofU.correctBoundaryConditions();
        auto& oldOfU = ofU.oldTime();
        oldOfU.primitiveFieldRef() = Foam::vector(0.0, 0.0, 0.0);
        oldOfU.correctBoundaryConditions();

        auto& nfU = nf::constructAndRegister(fieldCol, rt, ofU);
        auto& nfOldU = fvcc::oldTime(nfU);

        Foam::surfaceScalarField ofPhi(
            Foam::IOobject(
                "phi",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ),
            fvc::flux(ofU)
        );
        Foam::surfaceScalarField ofNu(
            Foam::IOobject(
                "nu",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar("nu", Foam::dimensionSet(0, 2, -1, 0, 0), 0.01)
        );

        auto nfPhi = nf::constructFrom(rt.exec, rt.nfMesh, ofPhi);
        auto nfNu = nf::constructFrom(rt.exec, rt.nfMesh, ofNu);

        NeoN::fill(nfOldU.internalVector(), NeoN::Vec3(0.0, 0.0, 0.0));
        nfOldU.correctBoundaryConditions();

        Foam::fvVectorMatrix ofUEqn(
            fvm::ddt(ofU) + fvm::div(ofPhi, ofU) - fvm::laplacian(ofNu, ofU)
        );

        nf::PDESolver<NeoN::Vec3> nfUEqn(
            dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
            nfU,
            rt
        );

        // Map the OpenFOAM solver controls and opt into the L1-scaled residual stop.
        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        auto mappedU = nf::mapFvSolution(solverDict.subDict("U"));
        mappedU.insert("l1ScaledResidual", std::string("true"));
        // This fixture's U dict has relTol=0 and tolerance=1e-18, below the L1 residual's
        // numerical floor, so the criterion can only stop at maxIter. Cap it so the test does
        // not grind 1000 iterations; the asserted INITIAL residual (at iteration 0) is unaffected.
        mappedU.subDict("criteria").insert("iteration", NeoN::label(20));
        solverDict.subDict("U") = mappedU;

        // OpenFOAM ground truth: initialResidual() is the per-component scaled L1 residual at
        // the starting field.
        Foam::SolverPerformance<Foam::vector> ofPerf = Foam::solve(ofUEqn);
        const Foam::vector ofInitRes = ofPerf.initialResidual();

        auto stats = nfUEqn.solve();
        REQUIRE(stats.entries.size() == 3);

        for (std::size_t cmpt = 0; cmpt < 3; ++cmpt)
        {
            const auto& entry = stats.entries[cmpt];
            // criterion is active: a finite, converged residual
            REQUIRE(entry.initResNorm > 0.0);
            REQUIRE(entry.numIter > 0);
            REQUIRE(entry.finalResNorm < entry.initResNorm);
            // parity with OpenFOAM's initial residual (per component)
            REQUIRE(
                entry.initResNorm == Catch::Approx(ofInitRes[cmpt]).epsilon(1e-4).margin(1e-10)
            );
        }
    }
}
