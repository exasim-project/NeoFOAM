
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"
#include "constrainHbyA.H"

namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace dsl = NeoN::dsl;
namespace nnfvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

using NeoFOAM::EqualsInternal;
using NeoFOAM::EqualsBoundary;

extern Foam::Time* timePtr; // A single time object

TEST_CASE("DistributedMomentum")
{
    SECTION("Parallel sanity check")
    {
        REQUIRE(Foam::Pstream::parRun());
        REQUIRE(Foam::Pstream::nProcs() == 3);
    }

    float epsilon = 1e-32;
    Foam::Time& runTime = *timePtr;
    // CPUExecutor only: this machine has a single GPU, so a multi-rank test must not place data on
    // the device (rank-to-GPU contention). Mirrors the rest of the distributed suite (e.g.
    // test_snGrad_distributed). rAU/HbyA now carry processor BCs (halo exchange) that a single-GPU
    // box cannot service across ranks.
    const NeoN::Executor exec = NeoN::CPUExecutor {};
    const std::string execName = "CPUExecutor";

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);

    auto ofU = randomVectorField(runTime, mesh, "U");
    auto ofp = randomScalarField(runTime, mesh, "p");
    ofp.correctBoundaryConditions();
    ofU.correctBoundaryConditions();
    auto& oldOfU = ofU.oldTime();
    oldOfU.primitiveFieldRef() = Foam::vector(0.0, 0.0, 0.0);
    oldOfU.correctBoundaryConditions();

    auto& vectorCollection = nnfvcc::VectorCollection::instance(rt.db, "VectorCollection");
    auto& nfP = NeoFOAM::constructAndRegister(vectorCollection, rt, ofp, false);

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

    SECTION("Solve transient momentum without grad(p) on " + execName)
    {
        Foam::fvVectorMatrix ofUEqn(
            fvm::ddt(ofU) + fvm::div(ofPhi, ofU) - fvm::laplacian(ofNu, ofU)
        );

        nf::PDESolver<NeoN::Vec3> nfUEqn(
            dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
            nfU,
            rt
        );

        NeoN::fill(nfUEqn.linearSystem().rhs(), NeoN::Vec3(0.0, 0.0, 0.0));

        // require fields to be initially the same
        // NOTE we skip comparing boundary values for now, since in distributed they have
        // different order
        REQUIRE_THAT(nfP, EqualsInternal(ofp, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfP.boundaryData(), EqualsBoundary(ofp, ApproxScalar(epsilon)));
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector(epsilon)));
        REQUIRE_THAT(nfU.boundaryData(), EqualsBoundary(ofU, ApproxVector(epsilon)));

        // TODO will be added by a separate PR
        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

        Foam::solve(ofUEqn);
        auto solverStatsDist = nfUEqn.solve();

        auto [numIterDist, initResNormDist, finalResNormDist, solveTimeDist] =
            solverStatsDist.entries[0];

        REQUIRE(numIterDist != 0);
        REQUIRE(initResNormDist != 0);
        REQUIRE(finalResNormDist < initResNormDist);
        REQUIRE_THAT(nfU, EqualsInternal(ofU, ApproxVector(1e-03)));
    }
}
