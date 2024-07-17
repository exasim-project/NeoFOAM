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

Communicator::bufferType* Communicator::findDuplexBuffer()
{
    for (auto it = buffers.begin(); it != buffers.end(); ++it)
        if (!it->isCommInit()) return &(*it);
    return nullptr;
}

Communicator::bufferType* Communicator::createNewDuplexBuffer()
{
    // determine buffer size.
    std::vector<size_t> rankSendSize(mpiEnviron_.usizeRank());
    std::vector<size_t> rankReceiveSize(mpiEnviron_.usizeRank());
    for (std::size_t rank = 0; rank < mpiEnviron_.usizeRank(); ++rank)
    {
        rankSendSize[rank] = std::ssize(sendMap_[rank]);
        rankReceiveSize[rank] = std::ssize(receiveMap_[rank]);
    }
    buffers.emplace_back(mpi::FullDuplexCommBuffer(mpiEnviron_, rankSendSize, rankReceiveSize));
    return &buffers.back();
}
};
