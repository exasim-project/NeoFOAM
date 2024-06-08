// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <mpi.h>

namespace NeoFOAM::core::mpi
{
// Send data to a specific rank
template<typename T>
void send(const T* data, int count, int rank, int tag, MPI_Comm comm)
{
    MPI_Send(data, count, MPI_BYTE, rank, tag, comm);
}

// Receive data from a specific rank
template<typename T>
void recv(
    T* data, int count, int rank, int tag, MPI_Comm comm, MPI_Status* status = MPI_STATUS_IGNORE
)
{
    MPI_Recv(data, count, MPI_BYTE, rank, tag, comm, status);
}

// Send data to all ranks
template<typename T>
void bcast(T* data, int count, int root, MPI_Comm comm)
{
    MPI_Bcast(data, count, MPI_BYTE, root, comm);
}

// Send data to all ranks
template<typename T>
void bcast(const T* data, int count, int root, MPI_Comm comm)
{
    MPI_Bcast(const_cast<T*>(data), count, MPI_BYTE, root, comm);
}

// Send data to all ranks
template<typename T>
void bcast(T& data, int root, MPI_Comm comm)
{
    MPI_Bcast(&data, sizeof(T), MPI_BYTE, root, comm);
}

// Send data to all ranks
template<typename T>
void bcast(const T& data, int root, MPI_Comm comm)
{
    MPI_Bcast(const_cast<T*>(&data), sizeof(T), MPI_BYTE, root, comm);
}

// Send data to all ranks
template<typename T>
void bcast(T* data, int count, MPI_Comm comm)
{
    MPI_Bcast(data, count, MPI_BYTE, 0, comm);
}

// Send data to all ranks
template<typename T>
void bcast(const T* data, int count, MPI_Comm comm)
{
    MPI_Bcast(const_cast<T*>(data), count, MPI_BYTE, 0, comm);
}

// Send data to all ranks
template<typename T>
void bcast(T& data, MPI_Comm comm)
{
    MPI_Bcast(&data, sizeof(T), MPI_BYTE, 0, comm);
}
}
