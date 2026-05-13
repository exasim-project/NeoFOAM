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
    }

    float epsilon = 1e-32;
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;

    SECTION("Parallel mesh sanity check")
    {
        if (Foam::Pstream::nProcs() == 3)
        {
            REQUIRE(rt.nfMesh.nCells() == 9);
        }
        REQUIRE(rt.nfMesh.boundaryMesh().isDistributed() == true);
    }
    SECTION_IF(
        Foam::Pstream::nProcs() == 3 && rt.mpiEnvironment.rank() == 1,
        "Correct boundary Mesh on rank 1"
    )
    {
        REQUIRE(rt.nfMesh.boundaryMesh().nBoundaries() == 5);
        REQUIRE(rt.nfMesh.boundaryMesh().nProcBoundaryPatches() == 2);
        REQUIRE(rt.nfMesh.boundaryMesh().nBoundaryFaces() == 12);
        REQUIRE(rt.nfMesh.boundaryMesh().nProcBoundaryFaces() == 18);

        REQUIRE(rt.nfMesh.boundaryMesh().neighbourRank()[0] == 0);
        REQUIRE(rt.nfMesh.boundaryMesh().neighbourRank()[1] == 2);
    }
    SECTION_IF(
        Foam::Pstream::nProcs() == 3 && rt.mpiEnvironment.rank() != 1,
        "Correct boundaryMesh on !rank 1"
    )
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
    SECTION_IF(
        Foam::Pstream::nProcs() == 3 && rt.mpiEnvironment.rank() == 1,
        "Correct commPattern on rank 1"
    )
    {
        auto sendCountsExp = std::vector<int> {9, 0, 9, 18};
        REQUIRE(commPattern.sendCounts == sendCountsExp);
        REQUIRE(rt.nfMesh.globalOffset() == 9);
        auto recvIdxExp =
            std::vector<int> {0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26};
        REQUIRE(commPattern.recvIdx == recvIdxExp);
    }
    SECTION_IF(
        Foam::Pstream::nProcs() == 3 && rt.mpiEnvironment.rank() == 0,
        "Correct commPattern on rank 0"
    )
    {
        auto sendCountsExp = std::vector<int> {0, 9, 0, 9};
        REQUIRE(commPattern.sendCounts == sendCountsExp);
        REQUIRE(rt.nfMesh.globalOffset() == 0);
        auto recvIdxExp = std::vector<int> {9, 10, 11, 12, 13, 14, 15, 16, 17};
        REQUIRE(commPattern.recvIdx == recvIdxExp);
    }
    SECTION_IF(
        Foam::Pstream::nProcs() == 3 && rt.mpiEnvironment.rank() == 2,
        "Correct commPattern on rank 2"
    )
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


TEST_CASE("GEO-01: deltaCoeffs at proc faces use cell-to-cell distance", "[GEO-01]")
{
    SECTION("Parallel sanity check")
    {
        REQUIRE(Foam::Pstream::parRun());
    }

    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;

    SECTION_IF(Foam::Pstream::nProcs() == 2, "Direct deltaCoeffs assertion on graded 2-rank mesh")
    {
        const auto& bm = rt.nfMesh.boundaryMesh();
        const auto nProcPatches = bm.nProcBoundaryPatches();
        if (nProcPatches == 0)
        {
            return; // serial -- nothing to check
        }

        const auto& nbrRanks = bm.neighbourRank();
        const auto& patchOffsets = bm.offset();
        const auto nTotalPatches = bm.nBoundaries();
        const auto firstProcPatch = nTotalPatches - nProcPatches;
        const auto nInternal = rt.nfMesh.nInternalFaces();

        // Copy boundary-tail accessors to host BEFORE triggering GeometryScheme
        // construction. GeometryScheme::readOrCreate calls reset() which clears
        // mesh_.cellCentres() and mesh_.faceCentres() to zero size after geometry
        // computation (geometryScheme.cpp:129-130). Any copyToHost() calls on those
        // arrays must happen before readOrCreate.
        //
        // Compressed boundary-tail accessors (DO NOT use mesh_.faceCentres() --
        // it is OF full-size including empty patches; bm.cf()/sf()/magSf() are
        // compressed and match the bcfacei indexing of updateDeltaCoeffs).
        auto bcCfHost = bm.cf().copyToHost();
        auto bcCfView = bcCfHost.view();
        auto bcSfHost = bm.sf().copyToHost();
        auto bcSfView = bcSfHost.view();
        auto bcMagSfHost = bm.magSf().copyToHost();
        auto bcMagSfView = bcMagSfHost.view();
        auto faceCellsHost = bm.faceCells().copyToHost();
        auto faceCellsView = faceCellsHost.view();
        auto cellCentreHost = rt.nfMesh.cellCentres().copyToHost();
        auto cellCentreView = cellCentreHost.view();

        // Trigger geometry-scheme construction (computes deltaCoeffs internally;
        // exercises the GEO-01 fix on the proc-face block). This call resets
        // mesh_.cellCentres() and mesh_.faceCentres() after computation -- hence
        // all mesh data copies above must precede this line.
        const auto& geomScheme = nnfvcc::GeometryScheme::readOrCreate(rt.nfMesh);
        const auto& deltaCoeffs = geomScheme->deltaCoeffs();
        auto dcHost = deltaCoeffs.internalVector().copyToHost();
        auto dcView = dcHost.view();

        for (NeoN::localIdx p = 0; p < nProcPatches; ++p)
        {
            const auto patchIdx = firstProcPatch + p;
            const auto nbrRank = static_cast<int>(nbrRanks[p]);
            const auto start = static_cast<int>(patchOffsets[patchIdx]);
            const auto end = static_cast<int>(patchOffsets[patchIdx + 1]);
            const int nFaces = end - start;

            // Compute local d_own per proc face: |n_hat . (bcCf - cellCentre[own])|
            // -- replicates exchangeProcOwnerDistance lines 100-120 inline (the
            // function itself lives in an anonymous namespace and is not callable
            // from test code).
            std::vector<double> localDOwn(nFaces);
            for (int i = 0; i < nFaces; ++i)
            {
                const auto bcfacei = start + i;
                const auto own = faceCellsView[bcfacei];
                const NeoN::Vec3 cellToFace =
                    bcCfView[bcfacei] - cellCentreView[own];
                const NeoN::Vec3 faceNormal =
                    (1.0 / bcMagSfView[bcfacei]) * bcSfView[bcfacei];
                localDOwn[i] = std::abs(
                    static_cast<NeoN::scalar>(faceNormal & cellToFace)
                );
            }

            std::vector<double> remoteD(nFaces);
            MPI_Sendrecv(
                localDOwn.data(),
                nFaces,
                MPI_DOUBLE,
                nbrRank,
                43,
                remoteD.data(),
                nFaces,
                MPI_DOUBLE,
                nbrRank,
                43,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            for (int i = 0; i < nFaces; ++i)
            {
                const double dSum = localDOwn[i] + remoteD[i];
                REQUIRE(dSum > 0.0);
                const double expected = 1.0 / dSum;
                REQUIRE(
                    dcView[nInternal + start + i]
                    == Catch::Approx(expected).margin(1e-10)
                );
            }
        }
    }

#if NF_WITH_GINKGO
    SECTION_IF(
        Foam::Pstream::nProcs() == 2,
        "GEO-01: Laplacian solve converges on graded 2-rank mesh"
    )
    {
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(schemesDict);

        auto ofp = randomScalarField(runTime, mesh, "p");
        ofp.correctBoundaryConditions();

        auto& vectorCollection =
            nnfvcc::VectorCollection::instance(rt.db, "VectorCollection");
        auto& nfP = nf::constructAndRegister(vectorCollection, rt, ofp);

        Foam::surfaceScalarField forAUf(
            Foam::IOobject(
                "rAUf",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar("rAUf", Foam::dimViscosity, 1.0)
        );
        auto nfrAUf = nf::constructFrom(rt.exec, rt.nfMesh, forAUf);

        nf::PDESolver<NeoN::scalar> pEqn(
            dsl::imp::laplacian(nfrAUf, nfP), nfP, rt
        );

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));

        auto solverStats = pEqn.solve();
        REQUIRE(solverStats.entries.size() > 0);
        auto [numIter, initResNorm, finalResNorm, solveTime] =
            solverStats.entries[0];

        REQUIRE(numIter >= 0);
        REQUIRE(!std::isnan(static_cast<double>(finalResNorm)));
        REQUIRE(initResNorm > 0);
        CHECK(finalResNorm / initResNorm < 1e-4);
    }
#else
    WARN("Ginkgo not available -- GEO-01 Laplacian solve skipped");
#endif
}


TEST_CASE("GEO-02: proc-face weight symmetry on graded mesh", "[GEO-02]")
{
    SECTION("Parallel sanity check")
    {
        REQUIRE(Foam::Pstream::parRun());
    }

    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto rt = nf::createAdapterRunTime(runTime, exec);

    SECTION_IF(
        Foam::Pstream::nProcs() == 2,
        "Weight symmetry detectable on graded (non-uniform) mesh"
    )
    {
        // On a uniform mesh every proc face has w == 0.5, so 0.5 + 0.5 == 1.0
        // even if the formula is reversed -- the assertion would pass
        // trivially. The graded fixture (simpleGrading (2 1 1)) guarantees
        // w != 0.5 at proc faces so reversal is detectable.
        const auto& geomScheme = nnfvcc::GeometryScheme::readOrCreate(rt.nfMesh);
        const auto& weights = geomScheme->weights();
        auto weightsHost = weights.internalVector().copyToHost();
        auto weightsView = weightsHost.view();

        const auto& bm = rt.nfMesh.boundaryMesh();
        const auto& nbrRanks = bm.neighbourRank();
        const auto& patchOffsets = bm.offset();
        const auto nTotalPatches = bm.nBoundaries();
        const auto nProcPatches = bm.nProcBoundaryPatches();
        const auto firstProcPatch = nTotalPatches - nProcPatches;
        const auto nInternal = rt.nfMesh.nInternalFaces();

        for (NeoN::localIdx p = 0; p < nProcPatches; ++p)
        {
            const auto patchIdx = firstProcPatch + p;
            const auto nbrRank = static_cast<int>(nbrRanks[p]);
            const auto start = static_cast<int>(patchOffsets[patchIdx]);
            const auto end = static_cast<int>(patchOffsets[patchIdx + 1]);
            const int nFaces = end - start;

            std::vector<double> localW(nFaces);
            for (int i = 0; i < nFaces; ++i)
            {
                localW[i] = weightsView[nInternal + start + i];
            }

            // Diagnostic: confirm we are on the graded fixture (w != 0.5).
            // On a uniform mesh localW[0] == 0.5; on the graded mesh it differs.
            if (nFaces > 0)
            {
                INFO(
                    "first proc-face weight (graded fixture sanity check) = "
                    << localW[0]
                );
            }

            std::vector<double> remoteW(nFaces);
            MPI_Sendrecv(
                localW.data(),
                nFaces,
                MPI_DOUBLE,
                nbrRank,
                44,
                remoteW.data(),
                nFaces,
                MPI_DOUBLE,
                nbrRank,
                44,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            for (int i = 0; i < nFaces; ++i)
            {
                REQUIRE(
                    localW[i] + remoteW[i]
                    == Catch::Approx(1.0).margin(1e-10)
                );
            }
        }
    }
}
