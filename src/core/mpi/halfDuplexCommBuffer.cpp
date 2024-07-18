// // SPDX-License-Identifier: MIT
// // SPDX-FileCopyrightText: 2023 NeoFOAM authors

// #include "NeoFOAM/core/mpi/halfDuplexCommBuffer.hpp"

// namespace NeoFOAM
// {

// namespace mpi
// {

// template<MemorySpace>
// bool HalfDuplexCommBuffer::isComplete()
// {
//     NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
//     int flag;
//     for (auto& request : request_)
//     {
//         int err = MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
//         NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Test failed.");
//         if (!flag) return false;
//     }
//     return static_cast<bool>(flag);
// }

// template<MemorySpace>
// void HalfDuplexCommBuffer::send()
// {
//     NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
//     NF_DEBUG_ASSERT(isComplete(), "Communication buffer is already active.");
//     for (auto rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
//     {
//         if (rankOffset_[rank + 1] - rankOffset_[rank] == 0) continue;
//         isend<char>(
//             rankBuffer_.data() + rankOffset_[rank],
//             rankOffset_[rank + 1] - rankOffset_[rank],
//             rank,
//             tag_,
//             mpiEnviron_.comm(),
//             &request_[rank]
//         );
//     }
// }

// template<MemorySpace>
// void HalfDuplexCommBuffer::receive()
// {
//     NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
//     NF_DEBUG_ASSERT(isComplete(), "Communication buffer is already active.");
//     for (auto rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
//     {
//         if (rankOffset_[rank + 1] - rankOffset_[rank] == 0) continue;
//         irecv<char>(
//             rankBuffer_.data() + rankOffset_[rank],
//             rankOffset_[rank + 1] - rankOffset_[rank],
//             rank,
//             tag_,
//             mpiEnviron_.comm(),
//             &request_[rank]
//         );
//     }
// }

// template<MemorySpace>
// void HalfDuplexCommBuffer::waitComplete()
// {
//     NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
//     while (!isComplete())
//     {
//         // todo deadlock prevention.
//         // wait for the communication to finish.
//     }
// }

// template<MemorySpace>
// void HalfDuplexCommBuffer::finaliseComm()
// {
//     NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
//     NF_DEBUG_ASSERT(isComplete(), "Cannot finalise while buffer is active.");
//     for (auto& request : request_)
//         NF_DEBUG_ASSERT(
//             request == MPI_REQUEST_NULL, "MPI_Request not null, communication not complete."
//         );
//     tag_ = -1;
//     commName_ = "unassigned";
// }

// }

// } // namespace NeoFoam
