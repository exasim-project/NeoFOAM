// SPDX-License-Identifier: Unlicense
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include "NeoFOAM/mesh/unstructured/communicator.hpp"

using namespace NeoFOAM;
using namespace NeoFOAM::mpi;

TEST_CASE("CommMap Initialization")
{

    SECTION("Default constructor")
    {
        RankNodeCommGraph comm_map;
        int rank;
        int size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        REQUIRE(comm_map.Rank() == rank);
        REQUIRE(comm_map.SizeRank() == size);
        auto sizes = comm_map.SizeBuffer();
        REQUIRE(sizes.first.size() == size);
        REQUIRE(sizes.second.size() == size);
        for (const auto& s : sizes.first)
        {
            REQUIRE(s == 0);
        }
        for (const auto& r : sizes.second)
        {
            REQUIRE(r == 0);
        }
    }

    SECTION("Custom MPI_Comm constructor")
    {
        MPI_Comm custom_comm;
        MPI_Comm_split(MPI_COMM_WORLD, 0, 0, &custom_comm);
        RankNodeCommGraph comm_map(custom_comm);
        int rank;
        int size;
        MPI_Comm_rank(custom_comm, &rank);
        MPI_Comm_size(custom_comm, &size);

        REQUIRE(comm_map.Rank() == rank);
        REQUIRE(comm_map.SizeRank() == size);
        auto sizes = comm_map.SizeBuffer();
        REQUIRE(sizes.first.size() == size);
        REQUIRE(sizes.second.size() == size);
        for (const auto& s : sizes.first)
        {
            REQUIRE(s == 0);
        }
        for (const auto& r : sizes.second)
        {
            REQUIRE(r == 0);
        }

        MPI_Comm_free(&custom_comm);
    }
}

TEST_CASE("CommMap Send/Receive Buffer Sizes")
{
    RankNodeCommGraph comm_map;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    SECTION("Empty buffers")
    {
        auto sizes = comm_map.SizeBuffer();
        REQUIRE(sizes.first.size() == size);
        REQUIRE(sizes.second.size() == size);
        for (const auto& s : sizes.first)
        {
            REQUIRE(s == 0);
        }
        for (const auto& r : sizes.second)
        {
            REQUIRE(r == 0);
        }
    }

    SECTION("Non-empty buffers")
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (size > 1)
        {
            // Let's assume rank 0 sends to rank 1
            if (rank == 0)
            {
                comm_map.send[1].push_back({0, 1});
            }
            if (rank == 1)
            {
                comm_map.receive[0].push_back({0, 1});
            }

            auto sizes = comm_map.SizeBuffer();
            if (rank == 0)
            {
                REQUIRE(sizes.first[1] == 1);
                REQUIRE(sizes.second[1] == 0);
            }
            if (rank == 1)
            {
                REQUIRE(sizes.first[0] == 0);
                REQUIRE(sizes.second[0] == 1);
            }
        }
        else
        {
            WARN("Not enough ranks to test non-empty buffers");
        }
    }
}
