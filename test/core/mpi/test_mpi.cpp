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
        NeoFOAM::scalar value = 1.0;
        reduceAllScalar(&value, NeoFOAM::ReduceOp::Max, MPI_COMM_WORLD);
    }
}
