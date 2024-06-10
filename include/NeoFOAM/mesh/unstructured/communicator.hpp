// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <vector>
#include <unordered_map>
#include <memory>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/mpi/operators.hpp"

namespace NeoFOAM
{


struct CommNode
{ // local index in this rank and global index in the total mesh.
    int local_idx;
    int global_idx;
};

struct CommMap
{ // This is a map of the nodes that need to be communicated from this rank to other ranks (and visa
    // versa).

    std::vector<std::vector<CommNode>> send;    // position here is the buffer position
    std::vector<std::vector<CommNode>> receive; // position here is the buffer position

    CommMap() : comm(MPI_COMM_WORLD) { InitRankInfo(); }

    CommMap(MPI_Comm commGroup) : comm(commGroup) { InitRankInfo(); }

    int Rank() const { return i_rank; }

    int SizeRank() const { return max_ranks; }

    std::pair<std::vector<std::size_t>, std::vector<std::size_t>> SizeBuffer() const
    {
        std::pair<std::vector<std::size_t>, std::vector<std::size_t>> sizes;
        for (auto rank = 0; rank < max_ranks; ++rank)
        {
            sizes.first.push_back(send[rank].size());
            sizes.second.push_back(receive[rank].size());
        }
        return sizes;
    }

private:

    MPI_Comm comm {MPI_COMM_NULL};
    int i_rank {-1};    // this rank
    int max_ranks {-1}; // ranks to communicate with.

    void InitRankInfo()
    {
        MPI_Comm_rank(comm, &i_rank);
        MPI_Comm_size(comm, &max_ranks);
        send.resize(max_ranks);
        receive.resize(max_ranks);
    }
};

struct SRBuffer
{                 // used to send and receive the actual data.
    int tag {-1}; // -1 indicates not used.
    std::unique_ptr<MPI_Request> request {nullptr};

    std::vector<std::unique_ptr<char>> send;    // I am assuming we are always on the CPU -
                                                // otherwise data containers need to change
    std::vector<std::unique_ptr<char>> receive; // I am assuming we are always on the CPU -
                                                // otherwise data containers need to change
    std::vector<std::size_t> sendDataSize;
    std::vector<std::size_t> receiveDataSize;


    void resize(std::size_t size)
    {
        NF_DEBUG_ASSERT(tag == -1, "Communication buffer currently in use.");
        if (send.size() < size) return; // to avoid reallocation, we never size down.
        send.resize(size);
        receive.resize(size);
        sendDataSize.resize(size);
        receiveDataSize.resize(size);
    }

    template<typename valueType>
    void resizeBuffers(std::vector<std::size_t> sendSize, std::vector<std::size_t> receiveSize)
    {

        NF_DEBUG_ASSERT(tag == -1, "Communication buffer currently in use.");
        NF_DEBUG_ASSERT(
            sendSize.size() == receiveSize.size(),
            "Different number of send and receive size, not supported."
        );
        NF_DEBUG_ASSERT(sendSize.size() == send.size(), "Need to resize number buffers first.");

        auto typeSize = sizeof(valueType);

        for (auto rank = 0; rank < sendSize.size(); ++rank)
        {
            std::size_t dataSize = sendSize[rank] * typeSize;
            if (sendDataSize[rank] < dataSize)
                continue; // to avoid reallocation, we never size down.
            else
            {
                sendDataSize[rank] = dataSize;
                send[rank].reset(new char[sendDataSize[rank]]);
            }

            dataSize = receiveSize[rank] * typeSize;
            if (receiveDataSize[rank] < dataSize)
                continue; // to avoid reallocation, we never size down.
            else
            {
                receiveDataSize[rank] = dataSize;
                receive[rank].reset(new char[receiveDataSize[rank]]);
            }
        }
    }

    template<typename valueType>
    void initSend(Field<valueType>& field, CommMap commMap)
    {
        for (auto rank = 0; rank < commMap.max_ranks; ++rank)
        {
            auto sendSize = commMap.send[rank].size();
            auto sendBuffer = reinterpret_cast<valueType*>(send[rank].get());
            for (auto data = 0; data < sendSize; ++data) // load data
            {
                auto local_idx = commMap.send[rank][data].local_idx;
                sendBuffer[data] = field(local_idx);
            }
            mpi::sendScalar<valueType>(send, sendSize, rank, tag, commMap.comm, request.get());
        }
    }

    template<typename valueType>
    void initRecieve(Field<valueType>& field, CommMap commMap)
    {
        for (auto rank = 0; rank < commMap.max_ranks; ++rank)
        {
            mpi::recvScalar<valueType>(
                receive, commMap.receive[rank].size(), rank, tag, commMap.comm, request.get()
            );
        }
    }

    template<typename valueType>
    void startComm(Field<valueType>& field, CommMap commMap)
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
    void copyTo(Field<valueType>& field, CommMap commMap)
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
    void finishComm(Field<valueType>& field, CommMap commMap)
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

// responsible for one neighbourhood send/recieve.
template<typename valueType>
class Communicator
{

    CommMap commMap;
    std::unordered_map<std::string, std::shared_ptr<SRBuffer>> communication;
    std::shared_ptr<SRBuffer> SRbuffers; // we own it here for persistence.
};


} // namespace NeoFOAM
