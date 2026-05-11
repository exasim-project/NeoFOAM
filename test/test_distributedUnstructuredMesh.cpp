// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include <mpi.h>

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
    //auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto [execName, exec] = GENERATE(
    std::pair<std::string, NeoN::Executor>{"CPUExecutor", NeoN::CPUExecutor{}}
);

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

    SECTION("Proc face weight symmetry")
    {
        // For every proc face shared by ranks A and B, the linear interpolation
        // weights must satisfy w_A + w_B = 1.  This invariant is broken when
        // communicateBoundaryData receives wrong displacements because proc
        // patches are not in ascending neighbour-rank order.  The fix in
        // collectProcPatchOffsets (basicGeometryScheme.cpp) sorts offsets by
        // neighbour rank before the MPI exchange.
        //
        // This test passes for the current 3-process simple-decomposition mesh
        // because the sort is a no-op (patches are already in ascending order).
        // It would FAIL on a mesh with non-ascending proc-patch rank order
        // (e.g. scotch decomposition with 4+ processes) without the sort fix.
        const auto& geomScheme = nnfvcc::GeometryScheme::readOrCreate(rt.nfMesh);
        const auto& weights = geomScheme->weights();
        auto weightsHost = weights.internalVector().copyToHost();
        auto weightsView = weightsHost.view();

        const auto& bMesh = rt.nfMesh.boundaryMesh();
        const auto& nbrRanks = bMesh.neighbourRank();
        const auto& patchOffsets = bMesh.offset();
        const auto nTotalPatches = bMesh.nBoundaries();
        const auto nProcPatches = bMesh.nProcBoundaryPatches();
        const auto firstProcPatch = nTotalPatches - nProcPatches;
        const auto nInternal = rt.nfMesh.nInternalFaces();

        for (NeoN::localIdx p = 0; p < nProcPatches; ++p)
        {
            const auto patchIdx = firstProcPatch + p;
            const auto nbrRank = static_cast<int>(nbrRanks[p]);
            const auto start = static_cast<int>(patchOffsets[patchIdx]);
            const auto end = static_cast<int>(patchOffsets[patchIdx + 1]);
            const auto nFaces = end - start;

            std::vector<double> localW(nFaces);
            for (int i = 0; i < nFaces; ++i)
                localW[i] = weightsView[nInternal + start + i];

            std::vector<double> remoteW(nFaces);
            MPI_Sendrecv(
                localW.data(),
                nFaces,
                MPI_DOUBLE,
                nbrRank,
                42,
                remoteW.data(),
                nFaces,
                MPI_DOUBLE,
                nbrRank,
                42,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            for (int i = 0; i < nFaces; ++i)
            {
                REQUIRE(localW[i] + remoteW[i] == Catch::Approx(1.0).margin(1e-10));
            }
        }
    }

    SECTION("Proc patch correctBoundaryConditions routes data to the right rank")
    {
        // Detects the proc-patch displacement bug in computeCommunicationPattern
        // / communicateBoundaryData. Each rank's MPI Alltoallv displacement
        // sdispl[r] must point at the offset of the local proc patch that
        // targets rank r — irrespective of mesh-order. The previous
        // implementation built sdispl as a running sum over sendCounts which
        // only happened to be correct when mesh-order matched ascending
        // neighbour-rank order, so on every other decomposition some patches
        // would be exchanged with the wrong rank.
        //
        // The test tags every owner cell with the encoding
        //   value = rank * RANK_STRIDE + localCellIdx
        // After correctBoundaryConditions, the proc-tail of
        // boundaryData().value() must satisfy
        //   floor(value / RANK_STRIDE) == nbrRank[procPatch]
        // for the patch the entry lives in. If the displacement bug is active,
        // some patches will hold values whose rank prefix points at a different
        // neighbour rank — caught by REQUIRE below.
        //
        // Generality: passes on any rank count where mesh-order matches ascending
        // neighbour-rank order on every rank (e.g. the existing 3-rank simple
        // decomposition). Fails on 4+ ranks with hierarchical / scotch
        // decompositions where mesh-order mismatches ascending order on at least
        // one rank — and fails the same way without the sort fix.
        constexpr NeoN::scalar RANK_STRIDE = 1.0e6;
        const int myRank = rt.mpiEnvironment.rank();

        // Set the OF volScalarField to the (rank, localCellIdx) signature, then
        // run OF's correctBoundaryConditions so processor patches carry the
        // matching ghost values from neighbour ranks. constructAndRegister then
        // mirrors that into the NeoN VolumeField (internal + boundary).
        auto fp = randomScalarField(runTime, mesh, "p");
        forAll(fp, celli)
        {
            fp[celli] = static_cast<NeoN::scalar>(myRank) * RANK_STRIDE
                      + static_cast<NeoN::scalar>(celli);
        }
        fp.correctBoundaryConditions();

        auto& vectorCollection = nnfvcc::VectorCollection::instance(rt.db, "VectorCollection");
        auto& nfField = NeoFOAM::constructAndRegister(vectorCollection, rt, fp);

        // Run NeoN's boundary correction — this is the path under test. The
        // processor BC repopulates boundaryData from local owner cells, then
        // communicateBoundaryData does MPI Alltoallv. After this, proc-tail
        // boundaryData entries should carry the neighbour rank's signature.
        nfField.correctBoundaryConditions();

        auto bcValueHost = nfField.boundaryData().value().copyToHost();
        auto bcView = bcValueHost.view();

        const auto& bm = rt.nfMesh.boundaryMesh();
        const auto totalPatches = bm.nBoundaries();
        const auto procPatchCount = bm.nProcBoundaryPatches();
        const auto firstProcPatch = totalPatches - procPatchCount;
        const auto& patchOffsets = bm.offset();
        const auto& nbrRanks = bm.neighbourRank();

        for (NeoN::localIdx p = 0; p < procPatchCount; ++p)
        {
            const auto patchIdx = firstProcPatch + p;
            const auto expectedNbrRank = static_cast<int>(nbrRanks[p]);
            const auto start = patchOffsets[patchIdx];
            const auto end = patchOffsets[patchIdx + 1];

            for (auto bf = start; bf < end; ++bf)
            {
                const auto received = bcView[bf];
                const int receivedRank =
                    static_cast<int>(std::floor(received / RANK_STRIDE + 0.5));
                INFO(
                    "myRank=" << myRank << " procPatchIdx=" << p
                              << " expectedNbrRank=" << expectedNbrRank
                              << " bf=" << bf << " received=" << received
                              << " receivedRank=" << receivedRank
                );
                REQUIRE(receivedRank == expectedNbrRank);
            }
        }
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

    SECTION("MPI-01: Alltoallv recv counts are independent of send counts")
    {
        // After MPI-01 fix, computeCommunicationPattern derives independent recvCounts
        // via MPI_Alltoall. Verify structural invariant: all received global cell indices
        // are non-negative (valid), and proc-patch ranks have a non-empty recvIdx.
        for (auto idx : commPattern.recvIdx)
        {
            REQUIRE(idx >= 0);
        }
        if (rt.nfMesh.boundaryMesh().nProcBoundaryPatches() > 0)
        {
            REQUIRE(commPattern.recvIdx.size() > 0);
        }
    }
}
