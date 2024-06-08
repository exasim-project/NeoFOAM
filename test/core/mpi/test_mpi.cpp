// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <catch2/catch_test_macros.hpp>
#include "NeoFOAM/core/mpi/operations.hpp"


TEST_CASE("Send and receive data to and from a specific rank", "[mpi]")
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    SECTION("subDict")
    {
        if (rank == 0)
        {
            int data = 42;
            NeoFOAM::core::mpi::send(&data, 1, 1, 0, MPI_COMM_WORLD);
        }
        else if (rank == 1)
        {
            int data;
            NeoFOAM::core::mpi::recv(&data, 1, 0, 0, MPI_COMM_WORLD);
            REQUIRE(data == 42);
        }
    }
}
