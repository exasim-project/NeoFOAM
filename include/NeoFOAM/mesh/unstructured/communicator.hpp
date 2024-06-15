// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <vector>
#include <unordered_map>
#include <memory>


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


template<typename valueType>
class Communicator
{
public:

    Communicator()
    {
        // determine buffer size.
        std::vector<std::size_t> rankSendSize(environ.sizeRank());
        std::vector<std::size_t> rankReceiveSize(environ.sizeRank());
        for (auto rank = 0; rank < environ.sizeRank(); ++rank)
        {
            rankSendSize[rank] = sendMap[rank].size();
            rankReceiveSize[rank] = receiveMap[rank].size();
        }
        duplexBuffer_ = DuplexCommBuffer(MPIEnviron_, rankSendSize, rankReceiveSize)
    };
    ~Communicator() = default;

    template<typename valueType>
    void startComm(
        Field<valueType>& field, const std::string& commName
    ) // key should be file and line number as string
    {
        auto iterBuff = findDuplexBuffer();
        if (iterBuff == duplexBuffer_.end())
            iterBuff =
                duplexBuffer_.emplace(key, DuplexCommBuffer(MPIEnviron_, sendMap_, receiveMap_));

        iterBuff.initComm<valueType>(commName);
        for (auto rank = 0; rank < MPIEnviron_.sizeRank(); ++rank)
        {
            auto rankBuffer = iterBuff.get<valueType>(rank);
            if (rankBuffer.size() == 0) continue; // we don't send to this rank.

            for (auto data = 0; data < sendMap_.size(); ++data)
                rankBuffer[data] = field(sendMap_[rank][data].local_idx);
        }
        iterBuff.startComm();
    }

    void isComplete(std::string commName) // key should be file and line number as string
    {
        NF_DEBUG_ASSERT(
            duplexBuffer_.find(commName) != duplexBuffer_.end(),
            "No communication buffer associated with key: " << commName
        );
        duplexBuffer_[commName].isComplete()
    }

    void finaliseComm(
        Field<valueType>& field, std::string commName
    ) // key should be file and line number as string
    {
        NF_DEBUG_ASSERT(
            duplexBuffer_.find(commName) != duplexBuffer_.end(),
            "No communication buffer associated with key: " << commName
        );

        duplexBuffer_[commName].waitComplete();
        for (auto rank = 0; rank < MPIEnviron_.sizeRank(); ++rank)
        {
            auto rankBuffer = iterBuff.get<valueType>(rank);
            if (rankBuffer.size() == 0) continue; // we don't send to this rank.

            for (auto data = 0; data < receiveMap_.size(); ++data)
                field(receiveMap_[rank][data].local_idx) = rankBuffer[data];
        }
        duplexBuffer_[commName].finaliseComm();
    }

private:

    mpi::MPIEnvironment MPIEnviron_;
    RankSimplexCommMap sendMap_;
    RankSimplexCommMap receiveMap_;
    std::unordered_map<std::string, DuplexCommBuffer> duplexBuffer_;

    std::unordered_map<std::string, DuplexCommBuffer>::iterator findDuplexBuffer()
    {
        for (auto it = duplexBuffer_.begin(); it != duplexBuffer_.end(); ++it)
            if (!it->isCommInit()) return it;
        return duplexBuffer_.end();
    }
};


} // namespace NeoFOAM
