// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <typeindex>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/mpi/buffer.hpp"
#include "NeoFOAM/core/mpi/operators.hpp"
#include "NeoFOAM/core/mpi/environment.hpp"

namespace NeoFOAM
{

/**
 * @brief Represents a pair of indices for local and global indexing.
 */
struct NodeCommMap
{
    int local_idx;  /**< The local index. */
    int global_idx; /**< The global index. */
};

using SimplexCommMap = std::vector<NodeCommMap>;

using RankSimplexCommMap = std::vector<SimplexCommMap>;

struct SimplexComm
{

    SimplexComm() = default;
    SimplexComm(const mpi::MPIEnvironment& mpiEnvr, const RankSimplexCommMap& CommMap)
        : mpiEnvr_(mpiEnvr), CommMap_(CommMap)
    {}

    ~SimplexComm() = default;

    template<typename valueType>
    void initSend(Field<valueType>& field, int tag)
    {
        tag_ = tag;
        Buffer_.setCommTypeSize(std::vector<std::size_t> rankComm); // todo
        for (auto rank = 0; rank < mpiEnvr_.sizeRank(); ++rank)
        {
            auto rankBuffer = Buffer_.get<valueType>(rank);
            for (auto data = 0; data < rankBuffer.size(); ++data)
                rankBuffer[data] = field(CommMap_[rank][data].local_idx);
            mpi::sendScalar<valueType>(
                rankBuffer[rank].data(), rankBuffer.size(), rank, tag_, mpiEnvr_.comm(), &request_
            );
        }
    }

    template<typename valueType>
    void initReceive(int tag)
    {
        tag_ = tag;
        rankBuffer.setCommTypeSize(std::vector<std::size_t> rankComm); // todo
        for (auto rank = 0; rank < mpiEnvr_.sizeRank(); ++rank)
        {
            auto rankBuffer = Buffer_.get<valueType>(rank);
            mpi::recvScalar<valueType>(
                receiveBuffer[rank].data(),
                rankBuffer.size(),
                rank,
                tag_,
                mpiEnvr_.comm(),
                &request_
            );
        }
    }

    bool test()
    {
        int flag;
        int err = MPI_Test(&request_, &flag, MPI_STATUS_IGNORE);
        NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Test failed.");
        return static_cast<bool>(flag);
    }

    template<typename valueType>
    void finishComm(Field<valueType>& field)
    {
        NF_DEBUG_ASSERT(tag != -1, "Communication buffer not in use.");
        while (!test())
        {
            // todo deadlock prevention.
            // wait for the communication to finish.
        }
        for (auto rank = 0; rank < mpiEnvr_.sizeRank(); ++rank)
        {
            auto rankBuffer = Buffer_.get<valueType>(rank);
            for (auto data = 0; data < rankBuffer.size(); ++data) // load data
                field(CommMap_[rank][data].local_idx) = rankBuffer[data];
        }
        tag_ = -1;
    }

private:

    const mpi::MPIEnvironment& mpiEnvr_;
    const RankSimplexCommMap& CommMap_;
    Buffer Buffer_;

    MPI_Request request_;
    int tag_;
};

struct DuplexBuffer
{
    SimplexComm send;
    SimplexComm receive;
};

template<typename valueType>
class Communicator
{
public:

    Communicator()
        : duplexBuffer_({SimplexComm(MPIEnviron_, sendMap_), SimplexComm(MPIEnviron_, receiveMap_)}
        ) {};
    ~Communicator() = default;

    template<typename valueType>
    void startComm(Field<valueType>& field, std::string key)
    {
        duplexBuffer[key].startComm(field, tag);
    }


private:

    mpi::MPIEnvironment MPIEnviron_;
    RankSimplexCommMap sendMap_;
    RankSimplexCommMap receiveMap_;
    DuplexBuffer duplexBuffer_;


    // template<typename valueType>
    // void initComm(Field<valueType>& field, RankNodeCommGraph commMap)
    // {
    //     for (auto rank = 0; rank < commMap.max_ranks; ++rank)
    //     {
    //         auto sendSize = commMap.sendMap[rank].size();
    //         auto sendBuffer = reinterpret_cast<valueType*>(send[rank].get());
    //         for (auto data = 0; data < sendSize; ++data) // load data
    //         {
    //             auto local_idx = commMap.sendMap[rank][data].local_idx;
    //             sendBuffer[data] = field(local_idx);
    //         }
    //         mpi::sendScalar<valueType>(send, sendSize, rank, tag, commMap.comm, request.get());
    //          mpi::recvScalar<valueType>(
    //             receive, commMap.receive[rank].size(), rank, tag, commMap.comm, request.get()
    //         );
    //     }
    // }

    // template<typename valueType>
    // void startComm(Field<valueType>& field, RankNodeCommGraph commMap)
    // {
    //     NF_DEBUG_ASSERT(tag == -1, "Communication buffer currently in use.");
    //     resize(commMap.max_ranks);
    //     auto [sendSize, receiveSize] = commMap.SizeBuffer();
    //     resizeBuffers<valueType>(sendSize, receiveSize);
    //     tag = 50;
    //     initSend(field, commMap);
    //     initRecieve(field, commMap);
    // }

    // bool test()
    // {
    //     int flag;
    //     int err = MPI_Test(request.get(), &flag, MPI_STATUS_IGNORE);
    //     NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Test failed.");
    //     return static_cast<bool>(flag);
    // }

    // template<typename valueType>
    // void copyTo(Field<valueType>& field, RankNodeCommGraph commMap)
    // {
    //     for (auto rank = 0; rank < commMap.max_ranks; ++rank)
    //     {
    //         auto recieveBuffer = reinterpret_cast<valueType*>(receive[rank].get());
    //         for (auto data = 0; data < commMap.send[rank].size(); ++data) // load data
    //         {
    //             auto local_idx = commMap.send[rank][data].local_idx;
    //             field(local_idx) = recieveBuffer[data];
    //         }
    //     }
    // }

    // template<typename valueType>
    // void finishComm(Field<valueType>& field, RankNodeCommGraph commMap)
    // {
    //     NF_DEBUG_ASSERT(tag != -1, "Communication buffer not in use.");
    //     while (!test())
    //     {
    //         // todo deadlock prevention.
    //         // wait for the communication to finish.
    //     }
    //     auto [sendSize, receiveSize] = commMap.SizeBuffer();
    //     copyTo(field, commMap);
    //     tag = -1;
    // }
};


} // namespace NeoFOAM
