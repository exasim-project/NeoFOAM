// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#include "NeoFOAM/mesh/unstructured/communicator.hpp"

namespace NeoFOAM
{

/**
 * @brief Checks if the communication with the given name is complete.
 * @param commName The communication name.
 */
void Communicator::isComplete(std::string commName)
{
    NF_DEBUG_ASSERT(
        CommBuffer_.find(commName) != CommBuffer_.end(),
        "No communication buffer associated with key: " << commName
    );
    CommBuffer_[commName].isComplete();
}


/**
 * @brief Finds an uninitialized communication buffer.
 * @return An iterator to the uninitialized communication buffer, or end iterator if not found.
 */
std::unordered_map<std::string, mpi::FullDuplexCommBuffer>::iterator
Communicator::findDuplexBuffer()
{
    for (auto it = CommBuffer_.begin(); it != CommBuffer_.end(); ++it)
        if (!it->second.isCommInit()) return it;
    return CommBuffer_.end();
}

/**
 * @brief Creates a new communication buffer with the given name.
 * @param commName The communication name.
 * @return An iterator to the newly created communication buffer.
 */
std::unordered_map<std::string, mpi::FullDuplexCommBuffer>::iterator
Communicator::createNewDuplexBuffer(std::string commName)
{
    // determine buffer size.
    std::vector<std::size_t> rankSendSize(MPIEnviron_.sizeRank());
    std::vector<std::size_t> rankReceiveSize(MPIEnviron_.sizeRank());
    for (auto rank = 0; rank < MPIEnviron_.sizeRank(); ++rank)
    {
        rankSendSize[rank] = sendMap_[rank].size();
        rankReceiveSize[rank] = receiveMap_[rank].size();
    }
    return CommBuffer_
        .emplace(commName, mpi::FullDuplexCommBuffer(MPIEnviron_, rankSendSize, rankReceiveSize))
        .first;
}
};
