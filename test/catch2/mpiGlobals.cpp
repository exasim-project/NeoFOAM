// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "mpiGlobals.hpp"

#include <chrono>
#include <thread>
#include <vector>


// Define MPI comm as global variable, since the Catch reporter can't be constructed
// with an MPI comm. Thus, to still access it, it's stored globally
MPI_Comm COMM = MPI_COMM_WORLD;

int ROOT = 0;
int RANK;
int COMM_SIZE;

bool IS_ROOT = false;

void serializeIO(volatile bool* threadShutdown)
{
    if (!IS_ROOT)
    {
        return;
    }

    std::vector<MPI_Request> req(COMM_SIZE);
    bool recvBuffer;
    for (int i = 0; i < COMM_SIZE; ++i)
    {
        MPI_Irecv(&recvBuffer, 1, MPI_CXX_BOOL, i, 666, COMM, &req[i]);
    }

    while (!*threadShutdown)
    {
        // completed will be the index within req of a request that has finished. This is
        // equivalent to rank of the process who wants to print
        int completed = MPI_UNDEFINED;
        int flag;
        while (completed == MPI_UNDEFINED && !*threadShutdown)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            MPI_Testany(req.size(), req.data(), &completed, &flag, MPI_STATUS_IGNORE);
        }

        if (*threadShutdown)
        {
            break;
        }

        // allow process to print
        const bool allowPrint = true;
        MPI_Send(&allowPrint, 1, MPI_CXX_BOOL, completed, 666, COMM);

        // wait until process has finished printing
        bool remoteHasFinished = false;
        MPI_Recv(&remoteHasFinished, 1, MPI_CXX_BOOL, completed, 666, COMM, MPI_STATUS_IGNORE);

        // re-activate request for process
        MPI_Irecv(&recvBuffer, 1, MPI_CXX_BOOL, completed, 666, COMM, &req[completed]);
    }
}
