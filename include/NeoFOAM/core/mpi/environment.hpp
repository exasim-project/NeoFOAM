// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <mpi.h>
#include "NeoFOAM/core/error.hpp"

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
        }

        /**
         * @brief Finalizes the MPI environment.
         */
        ~MPIEnvironment() { MPI_Finalize(); }

        /**
         * @brief Returns the rank of the current process.
         *
         * @return The rank of the current process.
         */
        int sizeRank() const { return m_rank; }

        /**
         * @brief Returns the number of ranks.
         *
         * @return The number of ranks.
         */
        int rank() const { return i_rank; }

        /**
         * @brief Returns the communicator.
         *
         * @return The communicator.
         */
        MPI_Comm comm() const { return communicator; }

    private:

        MPI_Comm communicator {MPI_COMM_NULL}; // MPI communicator
        int i_rank {-1};                       // Index of this rank
        int m_rank {-1};                       // Number of ranks in this communicator group.

        /**
         * @brief Updates the rank data, based on the communicator.
         */
        void updateRankData()
        {
            NF_ASSERT(communicator != MPI_COMM_NULL, "Invalid communicator, is null.");
            MPI_Comm_rank(communicator, &i_rank);
            MPI_Comm_size(communicator, &m_rank);
        }
    };

} // namespace mpi

} // namespace NeoFOAM
