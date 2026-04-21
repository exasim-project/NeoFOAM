
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
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;

    SECTION("Parallel mesh sanity check")
    {
        REQUIRE(rt.nfMesh.nCells() == 9);
        REQUIRE(rt.nfMesh.boundaryMesh().isDistributed() == true);
    }
    SECTION_IF(rt.mpiEnvironment.rank() == 1, "Correct boundary Mesh on rank 1")
    {
        REQUIRE(rt.nfMesh.boundaryMesh().nBoundaries() == 5);
        REQUIRE(rt.nfMesh.boundaryMesh().nProcBoundaryPatches() == 2);
        REQUIRE(rt.nfMesh.boundaryMesh().nBoundaryFaces() == 12);
        REQUIRE(rt.nfMesh.boundaryMesh().nProcBoundaryFaces() == 18);

        REQUIRE(rt.nfMesh.boundaryMesh().neighbourRank()[0] == 0);
        REQUIRE(rt.nfMesh.boundaryMesh().neighbourRank()[1] == 2);
    }
    SECTION_IF(rt.mpiEnvironment.rank() != 1, "Correct boundaryMesh on !rank 1")
    {
        REQUIRE(rt.nfMesh.boundaryMesh().nBoundaries() == 4);
        REQUIRE(rt.nfMesh.boundaryMesh().nProcBoundaryPatches() == 1);
        REQUIRE(rt.nfMesh.boundaryMesh().nBoundaryFaces() == 21);
        REQUIRE(rt.nfMesh.boundaryMesh().nProcBoundaryFaces() == 9);

        REQUIRE(rt.nfMesh.boundaryMesh().neighbourRank()[0] == 1);
    }

    auto commPattern = computeCommunicationPattern(rt.nfMesh);
    SECTION_IF(rt.mpiEnvironment.rank() == 1, "Correct commPattern on rank 1")
    {
        auto sendCountsExp = std::vector<int> {9, 0, 9, 18};
        REQUIRE(commPattern.sendCounts == sendCountsExp);
        REQUIRE(rt.nfMesh.globalOffset() == 9);
        auto recvIdxExp =
            std::vector<int> {0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26};
        REQUIRE(commPattern.recvIdx == recvIdxExp);
    }
    SECTION_IF(rt.mpiEnvironment.rank() == 0, "Correct commPattern on rank 0")
    {
        auto sendCountsExp = std::vector<int> {0, 9, 0, 9};
        REQUIRE(commPattern.sendCounts == sendCountsExp);
        REQUIRE(rt.nfMesh.globalOffset() == 0);
        auto recvIdxExp = std::vector<int> {9, 10, 11, 12, 13, 14, 15, 16, 17};
        REQUIRE(commPattern.recvIdx == recvIdxExp);
    }
    SECTION_IF(rt.mpiEnvironment.rank() == 2, "Correct commPattern on rank 2")
    {
        auto sendCountsExp = std::vector<int> {0, 9, 0, 9};
        REQUIRE(commPattern.sendCounts == sendCountsExp);
        REQUIRE(rt.nfMesh.globalOffset() == 18);
        auto recvIdxExp = std::vector<int> {9, 10, 11, 12, 13, 14, 15, 16, 17};
        REQUIRE(commPattern.recvIdx == recvIdxExp);
    }

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
        SECTION_IF(rt.mpiEnvironment.rank() == 0, "Correct fields on rank 0")
        {
            nf::compare(nfP, ofp, ApproxScalar(epsilon), false);
            nf::compare(nfU, ofU, ApproxVector(epsilon), false);
        }
        SECTION_IF(rt.mpiEnvironment.rank() == 1, "Correct fields on rank 1")
        {
            nf::compare(nfP, ofp, ApproxScalar(epsilon), false);
            nf::compare(nfU, ofU, ApproxVector(epsilon), false);
        }
        SECTION_IF(rt.mpiEnvironment.rank() == 2, "Correct fields on rank 2")
        {
            nf::compare(nfP, ofp, ApproxScalar(epsilon), false);
            nf::compare(nfU, ofU, ApproxVector(epsilon), false);
        }

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

        // Foam::solve(ofUEqn);
        auto solverStatsDist = nfUEqn.solve();

        auto [numIterDist, initResNormDist, finalResNormDist, solveTimeDist] =
            solverStatsDist.entries[0];

        REQUIRE(numIterDist != 0);
        REQUIRE(initResNormDist != 0);
        // nfU.correctBoundaryConditions();
        // nf::compare(nfU, ofU, ApproxVector({1e-08, 1e-08, 1e-08}));
    }
}
