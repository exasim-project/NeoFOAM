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

TEST_CASE("Distributed UnstructuredMesh")
{
    SECTION("Parallel sanity check")
    {
        REQUIRE(Foam::Pstream::parRun());
        REQUIRE(Foam::Pstream::nProcs() == 3);
    }

    float epsilon = 1e-32;
    Foam::Time& runTime = *timePtr;
    // CPU executor only for now; GPU MPI path requires CUDA-aware MPI.
    std::string execName = "CPUExecutor";
    NeoN::Executor exec = NeoN::CPUExecutor {};

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
}
