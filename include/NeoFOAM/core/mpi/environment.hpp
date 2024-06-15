// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <mpi.h>
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/info.hpp"

namespace NeoFOAM
{

namespace mpi
{

/**
 * @class MPIEnvironment
 * @brief A RAII class to manage MPI initialization and finalization with thread support.
 */
struct MPIEnvironment
{
    /**
     * @brief Initializes the MPI environment using a parsed communicator group.
     *
     * @param commGroup The communicator group, default is MPI_COMM_WORLD.
     */
    MPIEnvironment(MPI_Comm commGroup = MPI_COMM_WORLD) : communicator(commGroup)
    {
        updateRankData();
    }

    /**
     * @brief Initializes the MPI environment, ensuring thread support.
     *
     * @param argc Reference to the argument count.
     * @param argv Reference to the argument vector.
     */
    MPIEnvironment(int argc, char** argv)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        NF_ASSERT(
            provided == MPI_THREAD_MULTIPLE, "The MPI library does not have full thread support"
        );

        communicator = MPI_COMM_WORLD;
        updateRankData();
        isFinalize = true;
        NF_DINFO("MPI Rank: " << mpi_rank);
    }

    MPIEnvironment(const MPIEnvironment& other) noexcept
        : isFinalize(false), communicator(other.communicator), mpi_rank(other.mpi_rank),
          mpi_size(other.mpi_size)
    {}

    /**
     * @brief Finalizes the MPI environment.
     */
    ~MPIEnvironment()
    {
        if (isFinalize) MPI_Finalize();
    }

    /**
     * @brief Returns the number of ranks.
     *
     * @return The number of ranks.
     */
    int sizeRank() const { return mpi_size; }

    /**
     * @brief Returns the rank of the current process.
     *
     * @return The rank of the current process.
     */
    int rank() const { return mpi_rank; }

    /**
     * @brief Returns the communicator.
     *
     * @return The communicator.
     */
    MPI_Comm comm() const { return communicator; }

private:

    bool isFinalize {false};               // Flag to check if MPI_Finalize has been called.
    MPI_Comm communicator {MPI_COMM_NULL}; // MPI communicator
    int mpi_rank {-1};                     // Index of this rank
    int mpi_size {-1};                     // Number of ranks in this communicator group.

    /**
     * @brief Updates the rank data, based on the communicator.
     */
    void updateRankData()
    {
        NF_ASSERT(communicator != MPI_COMM_NULL, "Invalid communicator, is null.");
        MPI_Comm_rank(communicator, &mpi_rank);
        MPI_Comm_size(communicator, &mpi_size);
    }
};

} // namespace mpi

} // namespace NeoFOAM
