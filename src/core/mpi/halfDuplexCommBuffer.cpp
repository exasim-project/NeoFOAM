// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/mpi/halfDuplexCommBuffer.hpp"

namespace NeoFOAM
{

namespace mpi
{

bool HalfDuplexCommBuffer::isComplete()
{
    NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
    int flag;
    for (auto& request : request_)
    {
        int err = MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Test failed.");
        if (!flag) return false;
    }
    return static_cast<bool>(flag);
}

void HalfDuplexCommBuffer::send()
{
    NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
    NF_DEBUG_ASSERT(isComplete(), "Communication buffer is already active.");
    for (auto rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
    {
        if (rankOffset_[rank + 1] - rankOffset_[rank] == 0) continue;
        Isend<char>(
            rankBuffer_.data() + rankOffset_[rank],
            rankOffset_[rank + 1] - rankOffset_[rank],
            rank,
            tag_,
            mpiEnviron_.comm(),
            &request_[rank]
        );
    }
}

void HalfDuplexCommBuffer::receive()
{
    NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
    NF_DEBUG_ASSERT(isComplete(), "Communication buffer is already active.");
    for (auto rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
    {
        if (rankOffset_[rank + 1] - rankOffset_[rank] == 0) continue;
        Irecv<char>(
            rankBuffer_.data() + rankOffset_[rank],
            rankOffset_[rank + 1] - rankOffset_[rank],
            rank,
            tag_,
            mpiEnviron_.comm(),
            &request_[rank]
        );
    }
}

void HalfDuplexCommBuffer::waitComplete()
{
    NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
    while (!isComplete())
    {
        // todo deadlock prevention.
        // wait for the communication to finish.
    }
}

void HalfDuplexCommBuffer::finaliseComm()
{
    NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
    NF_DEBUG_ASSERT(isComplete(), "Cannot finalise while buffer is active.");
    for (auto& request : request_)
        NF_DEBUG_ASSERT(
            request == MPI_REQUEST_NULL, "MPI_Request not null, communication not complete."
        );
    tag_ = -1;
    commName_ = "unassigned";
}

}

} // namespace NeoFoam
