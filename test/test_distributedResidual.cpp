// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace dsl = NeoN::dsl;
namespace nnfvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

using NeoFOAM::EqualsInternal;
using NeoFOAM::EqualsBoundary;

extern Foam::Time* timePtr; // A single time object

// Verifies that NeoN's distributed L1-scaled residual stopping criterion reports the
// same (globally reduced, normFactor-scaled, L1) initial residual that OpenFOAM's
// distributed solverPerformance reports, and that it drives the solve to convergence.
TEST_CASE("DistributedL1Residual")
{
    SECTION("Parallel sanity check")
    {
        REQUIRE(Foam::Pstream::parRun());
        REQUIRE(Foam::Pstream::nProcs() == 3);
    }

    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);

    auto ofU = randomVectorField(runTime, mesh, "U");
    ofU.correctBoundaryConditions();
    auto& oldOfU = ofU.oldTime();
    oldOfU.primitiveFieldRef() = Foam::vector(0.0, 0.0, 0.0);
    oldOfU.correctBoundaryConditions();

    auto& vectorCollection = nnfvcc::VectorCollection::instance(rt.db, "VectorCollection");
    auto& nfU = NeoFOAM::constructAndRegister(vectorCollection, rt, ofU);
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

    auto nfPhi = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi);
    auto nfNu = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofNu);

    NeoN::fill(nfOldU.internalVector(), NeoN::Vec3(0.0, 0.0, 0.0));
    nfOldU.correctBoundaryConditions();

    SECTION("Distributed L1 residual matches OpenFOAM on " + execName)
    {
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
        solverDict.subDict("U") = mappedU;

        // OpenFOAM ground truth: its default residual norm is the L1-scaled norm, reduced
        // globally across ranks. initialResidual() is the per-component scaled residual at
        // the starting field.
        Foam::SolverPerformance<Foam::vector> ofPerf = Foam::solve(ofUEqn);
        const Foam::vector ofInitRes = ofPerf.initialResidual();

        auto stats = nfUEqn.solve();
        REQUIRE(stats.entries.size() == 3);

        for (std::size_t cmpt = 0; cmpt < 3; ++cmpt)
        {
            const auto& entry = stats.entries[cmpt];
            // criterion is active: a finite, reduced, converged residual
            REQUIRE(entry.initResNorm > 0.0);
            REQUIRE(entry.numIter > 0);
            REQUIRE(entry.finalResNorm < entry.initResNorm);
            // parity with OpenFOAM's distributed initial residual (per component)
            REQUIRE(
                entry.initResNorm == Catch::Approx(ofInitRes[cmpt]).epsilon(1e-4).margin(1e-10)
            );
        }
    }
}
