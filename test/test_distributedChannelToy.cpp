// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// Minimal sanity test for the doubly-graded toy reproducer case
// (test/setup_distributedChannelToy_mpi2/, see
// .planning/debug/gpu-proc-boundary-divergence.md track T2).
//
// The actual GPU-vs-CPU divergence hunt is driven by running the `neoIcoFoam`
// binary against this case directory with NEOFOAM_PROC_DUMP=1 / NEOFOAM_FULL_DUMP=1
// — this CTest entry just validates that the case decomposes correctly into 2
// ranks and that the proc-boundary geometry is set up as expected:
//
//   ranks                   : 2
//   total cells (global)    : 10 x 5 x 3 = 150
//   per-rank cells          : 75 each (hierarchical (2 1 1) split at x=0.5)
//   non-proc patches        : inlet, outlet, walls (frontAndBack=empty drops)
//   proc patches per rank   : 1 (neighbour is the other rank)
//
// Run:
//   mpirun -np 2 build/develop/bin/tests/neofoam_test_distributedChannelToy -parallel
// (from test/setup_distributedChannelToy_mpi2/ once blockMesh+decomposePar have run)

#define CATCH_CONFIG_RUNNER

#include <mpi.h>

#include "common.hpp"

namespace nf = NeoFOAM;

extern Foam::Time* timePtr;

TEST_CASE("Distributed channel toy mesh sanity")
{
    SECTION("Parallel sanity check")
    {
        REQUIRE(Foam::Pstream::parRun());
        REQUIRE(Foam::Pstream::nProcs() == 2);
    }

    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);

    SECTION("Per-rank cell count [" + execName + "]")
    {
        // hierarchical (2 1 1) splits the 10-cell streamwise direction at the
        // mid-point: cells [0..4] go to rank 0, cells [5..9] go to rank 1
        // (uniformly in y and z). Each rank therefore owns 5 x 5 x 3 = 75 cells.
        REQUIRE(rt.nfMesh.nCells() == 75);
    }

    SECTION("Distributed boundary mesh [" + execName + "]")
    {
        REQUIRE(rt.nfMesh.boundaryMesh().isDistributed());
        REQUIRE(rt.nfMesh.boundaryMesh().nProcBoundaryPatches() == 1);
        REQUIRE(rt.nfMesh.boundaryMesh().nProcBoundaryFaces() == 15);
    }

    SECTION("Proc-cut neighbour rank [" + execName + "]")
    {
        // Rank 0's only proc patch points at rank 1, and vice-versa.
        const int myRank = rt.mpiEnvironment.rank();
        const int otherRank = (myRank == 0) ? 1 : 0;
        REQUIRE(rt.nfMesh.boundaryMesh().neighbourRank()[0] == otherRank);
    }
}
