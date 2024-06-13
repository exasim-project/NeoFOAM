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
    {
        NF_ASSERT(
            CommMap_.size() == mpiEnvr_.sizeRank(), "Invalid rank vs. communication map size."
        );

        // determine buffer size.
        std::vector<std::size_t> rankCommSize(mpiEnvr_.sizeRank());
        for (auto rank = 0; rank < CommMap.size(); ++rank)
            rankCommSize[rank] = CommMap_[rank].size();
        Buffer_.setCommTypeSize<double>(rankCommSize); //
    }

    ~SimplexComm() = default;

    int tag() const { return tag_; }

    template<typename valueType>
    void initSend(Field<valueType>& field, int tag)
    {
        NF_DEBUG_ASSERT(tag_ == -1, "Communication buffer in use.");
        tag_ = tag;
        Buffer_.setCommType<valueType>();
        for (auto rank = 0; rank < mpiEnvr_.sizeRank(); ++rank)
        {
            auto rankBuffer = Buffer_.get<valueType>(rank);
            if (rankBuffer.size() == 0) continue; // we don't send to this rank.

            for (auto data = 0; data < rankBuffer.size(); ++data)
                rankBuffer[data] = copyField(CommMap_[rank][data].local_idx);
            mpi::sendScalar<valueType>(
                rankBuffer.data(), rankBuffer.size(), rank, tag_, mpiEnvr_.comm(), &request_
            );
        }
    }

    template<typename valueType>
    void initReceive(int tag)
    {
        NF_DEBUG_ASSERT(tag_ == -1, "Communication buffer in use.");
        tag_ = tag;
        Buffer_.setCommType<valueType>();
        for (auto rank = 0; rank < mpiEnvr_.sizeRank(); ++rank)
        {
            auto rankBuffer = Buffer_.get<valueType>(rank);
            if (rankBuffer.size() == 0) continue; // we don't receive form this rank.
            mpi::recvScalar<valueType>(
                rankBuffer.data(), rankBuffer.size(), rank, tag_, mpiEnvr_.comm(), &request_
            );
        }
    }

    bool test()
    {
        NF_DEBUG_ASSERT(tag_ != -1, "Communication buffer not in use.");
        int flag;
        int err = MPI_Test(&request_, &flag, MPI_STATUS_IGNORE);
        NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Test failed.");
        return static_cast<bool>(flag);
    }

    template<typename valueType>
    void finishComm(Field<valueType>& field)
    {
        NF_DEBUG_ASSERT(tag_ != -1, "Communication buffer not in use.");
        while (!test())
        {
            // todo deadlock prevention.
            // wait for the communication to finish.
        }
        for (auto rank = 0; rank < mpiEnvr_.sizeRank(); ++rank)
        {
            auto rankBuffer = Buffer_.get<valueType>(rank);
            for (auto data = 0; data < rankBuffer.size(); ++data) // load data
                field(CommMap_[rank][data].local_idx) =
                    rankBuffer[data]; // how do I copy back to device?
        }
        tag_ = -1;
    }

private:

    const mpi::MPIEnvironment& mpiEnvr_;
    const RankSimplexCommMap& CommMap_;
    Buffer Buffer_;

    MPI_Request request_;
    int tag_ {-1};
};

struct DuplexCommBuffer
{
    SimplexComm send_;
    SimplexComm receive_;

    DuplexCommBuffer() = default;
    DuplexCommBuffer(
        mpi::MPIEnvironment environ, RankSimplexCommMap sendMap, RankSimplexCommMap receiveMap
    )
        : send_(environ, sendMap), receive_(environ, receiveMap)
    {}
};

int bufferHash(const std::string& key)
{
    std::hash<std::string> hash_fn; // I know its not completely safe, but it will do for now.
    return static_cast<int>(hash_fn(key));
}

template<typename valueType>
class Communicator
{
public:

    Communicator() : duplexBuffer_(MPIEnviron_, sendMap_, receiveMap_) {};
    ~Communicator() = default;

    template<typename valueType>
    void startComm(
        Field<valueType>& field, std::string key
    ) // key should be file and line number as string
    {
        auto iterBuff = findDuplexBuffer();
        if (iterBuff == duplexBuffer_.end())
            iterBuff =
                duplexBuffer_.emplace(key, DuplexCommBuffer(MPIEnviron_, sendMap_, receiveMap_));

        duplexBuffer[key].receive_.initReceive(bufferHash(key));
        duplexBuffer[key].send_.initSend(field, bufferHash(key));
    }

    void test(std::string key) // key should be file and line number as string
    {
        duplexBuffer_[key].receive_.test();
        duplexBuffer_[key].send_.test();
    }

    void finishComm(
        Field<valueType>& field, std::string key
    ) // key should be file and line number as string
    {
        duplexBuffer_[key].receive_.finishComm(field);
    }

private:

    mpi::MPIEnvironment MPIEnviron_;
    RankSimplexCommMap sendMap_;
    RankSimplexCommMap receiveMap_;
    std::unordered_map<std::string, DuplexCommBuffer> duplexBuffer_;


    std::unordered_map<std::string, DuplexCommBuffer>::iterator findDuplexBuffer()
    {
        for (auto it = duplexBuffer_.begin(); it != duplexBuffer_.end(); ++it)
            if (it->second.send.tag() == -1 && it->second.receive.tag() == -1) return it;
        return duplexBuffer_.end();
    }
};


} // namespace NeoFOAM
