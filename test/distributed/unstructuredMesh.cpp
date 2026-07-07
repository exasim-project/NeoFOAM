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
}
