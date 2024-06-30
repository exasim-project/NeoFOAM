// SPDX-License-Identifier: Unlicense
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#include <source_location>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/mesh/unstructured/communicator.hpp"
#include "NeoFOAM/core/mpi/environment.hpp"
#include "NeoFOAM/fields/field.hpp"

using namespace NeoFOAM;

TEST_CASE("Communicator Field Synchronization")
{
    mpi::MPIEnvironment MPIEnviron;
    Communicator comm;

    // first block send (size rank)
    // second block remains the same
    // third block receive (size rank)
    Field<int> field(CPUExecutor(), 3 * MPIEnviron.sizeRank());

    for (int rank = 0; rank < MPIEnviron.sizeRank(); rank++)
    {
        // we send the rank numbers
        field(rank) = rank;

        // just make sure its not a communicated value.
        field(rank + MPIEnviron.sizeRank()) = MPIEnviron.sizeRank() + rank;

        // set to 0.0 to check if the value is communicated
        field(rank + 2 * MPIEnviron.sizeRank()) = 0;
    }

    // Set up buffer to local map, we will ignore global_idx
    CommMap rankSendMap(MPIEnviron.sizeRank());
    CommMap rankReceiveMap(MPIEnviron.sizeRank());
    for (int rank = 0; rank < MPIEnviron.sizeRank(); rank++)
    {
        rankSendMap[rank].emplace_back(NodeCommMap {.local_idx = rank});
        NodeCommMap newNode({.local_idx = 2 * MPIEnviron.sizeRank() + rank});
        rankReceiveMap[rank].push_back(newNode); // got tired of fighting with clang-format.
    }

    // Communicate
    comm = Communicator(MPIEnviron, rankSendMap, rankReceiveMap);
    std::string loc =
        std::source_location::current().file_name() + std::source_location::current().line();
    comm.startComm(field, loc);
    comm.isComplete(loc); // just call it to make sure it doesn't crash
    comm.finaliseComm(field, loc);

    // Check the values
    for (int rank = 0; rank < MPIEnviron.sizeRank(); rank++)
    {
        REQUIRE(field(rank) == rank);
        REQUIRE(field(rank + MPIEnviron.sizeRank()) == MPIEnviron.sizeRank() + rank);
        REQUIRE(field(rank + 2 * MPIEnviron.sizeRank()) == MPIEnviron.rank());
    }
}
