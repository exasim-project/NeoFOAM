// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <typeindex>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/mpi/operators.hpp"
#include "NeoFOAM/core/mpi/environment.hpp"

namespace NeoFOAM
{

/**
 * @brief Represents a pair of indices for local and global indexing.
 */
struct CommNode
{
    int local_idx;  /**< The local index. */
    int global_idx; /**< The global index. */
};

/**
 * @brief Represents a map of the nodes that need to be communicated from this rank to other ranks
 * (and vice versa).
 *
 * The `RankNodeCommGraph` class provides functionality for managing the communication of nodes
 * (cells) between ranks. It contains maps which are used to determine to which rank and node, a
 * node in this rank is communicating, either sending or recieving. The position in this class
 * structure also implicitly gives the position in the buffer size (they should be 1:1).
 *
 * The class also provides methods for retrieving the rank, size of the rank, and MPI communicator
 * associated with the communication environment. Additionally, it provides a method for obtaining
 * the sizes of the send and receive buffers for each rank.
 *
 * @note This class requires the `mpi::MPIEnvironment` class for MPI communication.
 */
class RankNodeCommGraph
{
public:

    RankNodeCommGraph();

    void resizeRank(int size)
    {
        sendMap.resize(size);
        receiveMap.resize(size);
    };

    std::pair<std::vector<std::size_t>, std::vector<std::size_t>> SizeBuffer() const
    {
        std::pair<std::vector<std::size_t>, std::vector<std::size_t>> sizes;
        for (auto rank = 0; rank < sizes.first.size(); ++rank)
            sizes.first.push_back(sendMap[rank].size());
        for (auto rank = 0; rank < sizes.second.size(); ++rank)
            sizes.second.push_back(receiveMap[rank].size());
        return sizes;
    }

private:

    std::vector<std::vector<CommNode>> sendMap;
    std::vector<std::vector<CommNode>> receiveMap;
};

template<typename valueType>
class Communicator
{

    mpi::MPIEnvironment commEnvr;
    RankNodeCommGraph commMap;
    std::unordered_map<std::string, std::shared_ptr<SRBuffer>> communication;
    std::shared_ptr<SRBuffer> SRbuffers; // we own it here for persistence.

    // used to send and receive the actual data.
    int tag {-1}; // -1 indicates not used.
    std::unique_ptr<MPI_Request> request {nullptr};

    template<typename valueType>
    void initSend(Field<valueType>& field, RankNodeCommGraph commMap)
    {
        for (auto rank = 0; rank < commMap.max_ranks; ++rank)
        {
            auto sendSize = commMap.sendMap[rank].size();
            auto sendBuffer = reinterpret_cast<valueType*>(send[rank].get());
            for (auto data = 0; data < sendSize; ++data) // load data
            {
                auto local_idx = commMap.sendMap[rank][data].local_idx;
                sendBuffer[data] = field(local_idx);
            }
            mpi::sendScalar<valueType>(send, sendSize, rank, tag, commMap.comm, request.get());
        }
    }

    template<typename valueType>
    void initRecieve(Field<valueType>& field, RankNodeCommGraph commMap)
    {
        for (auto rank = 0; rank < commMap.max_ranks; ++rank)
        {
            mpi::recvScalar<valueType>(
                receive, commMap.receive[rank].size(), rank, tag, commMap.comm, request.get()
            );
        }
    }

    template<typename valueType>
    void startComm(Field<valueType>& field, RankNodeCommGraph commMap)
    {
        NF_DEBUG_ASSERT(tag == -1, "Communication buffer currently in use.");
        resize(commMap.max_ranks);
        auto [sendSize, receiveSize] = commMap.SizeBuffer();
        resizeBuffers<valueType>(sendSize, receiveSize);
        tag = 50;
        initSend(field, commMap);
        initRecieve(field, commMap);
    }

    bool test()
    {
        int flag;
        int err = MPI_Test(request.get(), &flag, MPI_STATUS_IGNORE);
        NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Test failed.");
        return static_cast<bool>(flag);
    }

    template<typename valueType>
    void copyTo(Field<valueType>& field, RankNodeCommGraph commMap)
    {
        for (auto rank = 0; rank < commMap.max_ranks; ++rank)
        {
            auto recieveBuffer = reinterpret_cast<valueType*>(receive[rank].get());
            for (auto data = 0; data < commMap.send[rank].size(); ++data) // load data
            {
                auto local_idx = commMap.send[rank][data].local_idx;
                field(local_idx) = recieveBuffer[data];
            }
        }
    }

    template<typename valueType>
    void finishComm(Field<valueType>& field, RankNodeCommGraph commMap)
    {
        NF_DEBUG_ASSERT(tag != -1, "Communication buffer not in use.");
        while (!test())
        {
            // todo deadlock prevention.
            // wait for the communication to finish.
        }
        auto [sendSize, receiveSize] = commMap.SizeBuffer();
        copyTo(field, commMap);
        tag = -1;
    }
};


} // namespace NeoFOAM
