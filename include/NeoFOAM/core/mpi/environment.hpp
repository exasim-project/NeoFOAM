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
        }

        /**
         * @brief Finalizes the MPI environment.
         */
        ~MPIEnvironment() { MPI_Finalize(); }
    };

} // namespace mpi

} // namespace NeoFOAM
