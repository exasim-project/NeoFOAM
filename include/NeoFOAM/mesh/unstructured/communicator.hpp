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
struct NodeMap
{
    int local_idx;  /**< The local index. */
    int global_idx; /**< The global index. */
};

struct DomainCommMap
{
    std::vector<std::vector<NodeMap>> commNodes; /**< The nodes that need to be communicated. */
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

    std::vector<std::vector<NodeMap>> sendMap;
    std::vector<std::vector<NodeMap>> receiveMap;
};


/**
 * @class Buffer
 * @brief A class that represents a data buffer for communication in a distributed system.
 *
 * The Buffer class provides functionality for managing the data buffer for communication between
 * ranks in memory efficient and type flexible way. To prevent multiple reallocations, the buffer
 * size is never sized down.
 *
 * The maximum buffer size is determined by:
 * size = sum_r sum_cn sizeof(valueType)
 */
class Buffer
{

public:

    Buffer() {}
    ~Buffer() {}

    /**
     * @brief Set the buffer data size based on the rank communication and value type.
     *
     * @tparam valueType The type of the data to be stored in the buffer.
     * @param rankComm The number of nodes to be communicated to each rank.
     */
    template<typename valueType>
    void setCommTypeSize(std::vector<std::size_t> rankComm)
    {
        dataType = typeid(valueType);
        std::size_t dataSize = 0;
        rankOffset.resize(rankComm.size() + 1);

        for (auto rank = 0; rank < rankComm.size(); ++rank)
        {
            rankOffset[rank] = dataSize;
            dataSize += rankComm[rank] * sizeof(valueType);
        }
        rankOffset.back() = dataSize;

        if (rankBuffer < dataSize) rankBuffer.resize(dataSize); // we never size down.
    }

    /**
     * @brief Get a span of the buffer data for a given rank.
     *
     * @tparam valueType The type of the data to be stored in the buffer.
     * @param rank The rank of the data to be retrieved.
     * @return std::span<valueType> A span of the data for the given rank.
     */
    template<typename valueType>
    std::span<valueType> get(const int rank)
    {
        NF_DEBUG_ASSERT(dataType == typeid(valueType), "Data type mismatch.");
        return std::span<valueType>(
            reinterpret_cast<valueType*>(rankBuffer.data() + rankOffset[rank]),
            rankComm[rank + 1] - rankComm[rank]
        );
    }

    /**
     * @brief Get a span of the buffer data for a given rank.
     *
     * @tparam valueType The type of the data to be stored in the buffer.
     * @param rank The rank of the data to be retrieved.
     * @return std::span<const valueType> A span of the data for the given rank.
     */
    template<typename valueType>
    std::span<const valueType> get(const int rank) const
    {
        NF_DEBUG_ASSERT(dataType == typeid(valueType), "Data type mismatch.");
        return std::span<const valueType>(
            reinterpret_cast<valueType*>(rankBuffer.data() + rankOffset[rank]),
            rankComm[rank + 1] - rankComm[rank]
        );
    }

private:

    std::type_index dataType {typeid(char)}; /*< The data type currently stored in the buffer. */
    std::vector<char> rankBuffer;            /*< The buffer data for all ranks. Never shrinks. */
    std::vector<std::size_t> rankOffset;     /*< The offset for a rank data in the buffer. */
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
