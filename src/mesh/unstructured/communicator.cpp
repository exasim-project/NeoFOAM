// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#include "NeoFOAM/mesh/unstructured/communicator.hpp"

namespace NeoFOAM
{

bool Communicator::isComplete(std::string commName)
{
    NF_DEBUG_ASSERT(
        CommBuffer_.find(commName) != CommBuffer_.end(),
        "No communication buffer associated with key: " << commName
    );
    return CommBuffer_[commName]->isComplete();
}

bufferType* Communicator::findDuplexBuffer()
{
    for (auto it = buffers.begin(); it != buffers.end(); ++it)
        if (!it->isCommInit()) return &(*it);
    return nullptr;
}

bufferType* Communicator::createNewDuplexBuffer()
{
    // determine buffer size.
    std::vector<std::size_t> rankSendSize(MPIEnviron_.sizeRank());
    std::vector<std::size_t> rankReceiveSize(MPIEnviron_.sizeRank());
    for (auto rank = 0; rank < MPIEnviron_.sizeRank(); ++rank)
    {
        rankSendSize[rank] = sendMap_[rank].size();
        rankReceiveSize[rank] = receiveMap_[rank].size();
    }
    buffers.emplace_back(mpi::FullDuplexCommBuffer(MPIEnviron_, rankSendSize, rankReceiveSize));
    return &buffers.back();
}
};
