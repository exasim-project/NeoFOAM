// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <vector>
#include <unordered_map>
#include <memory>


#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/mpi/fullDuplexCommBuffer.hpp"
#include "NeoFOAM/core/mpi/operators.hpp"
#include "NeoFOAM/core/mpi/environment.hpp"

namespace NeoFOAM
{

/**
 * @brief Represents a pair of indices for rank local and global indexing.
 */
struct NodeCommMap
{
    label local_idx;  /**< The local index. */
    label global_idx; /**< The global index. */
};

/**
 * @brief Represents a mapping of NodeCommMap for a rank.
 */
using SimplexCommMap = std::vector<NodeCommMap>;

/**
 * @brief Represents a mapping of SimplexCommMap for each rank.
 */
using RankSimplexCommMap = std::vector<SimplexCommMap>;

/**
 * @class Communicator
 * @brief Manages communication between ranks in a parallel environment.
 * The Communicator class provides functionality to manage communication of field data exchange
 * between MPI ranks for unstructured meshes. The class maintains an MPI environment and maps for
 * rank-specific send and receive operations.
 */
class Communicator
{
public:

    /**
     * @brief Default constructor.
     */
    Communicator() = default;

    /**
     * @brief Destructor.
     */
    ~Communicator() = default;

    /**
     * @brief Constructor that initializes the Communicator with MPI environment, rank send map, and
     * rank receive map.
     * @param MPIEnviron The MPI environment.
     * @param rankSendMap The rank send map.
     * @param rankReceiveMap The rank receive map.
     */
    Communicator(
        mpi::MPIEnvironment MPIEnviron,
        RankSimplexCommMap rankSendMap,
        RankSimplexCommMap rankReceiveMap
    )
        : MPIEnviron_(MPIEnviron), sendMap_(rankSendMap), receiveMap_(rankReceiveMap)
    {
        NF_DEBUG_ASSERT(
            MPIEnviron_.sizeRank() == rankSendMap.size(),
            "Size of rankSendSize does not match MPI size."
        );
        NF_DEBUG_ASSERT(
            MPIEnviron_.sizeRank() == rankReceiveMap.size(),
            "Size of rankReceiveSize does not match MPI size."
        );
    };

    /**
     * @brief Starts the non-blocking communication for a given field and communication name.
     * @tparam valueType The value type of the field.
     * @param field The field to be communicated/synchronised.
     * @param commName The communication name, typically a file and line number.
     */
    template<typename valueType>
    void startComm(Field<valueType>& field, const std::string& commName)
    {
        auto iterBuff = findDuplexBuffer();
        if (iterBuff == CommBuffer_.end()) iterBuff = createNewDuplexBuffer(commName);
        auto& buffer = (*iterBuff).second;

        buffer.initComm<valueType>(commName);
        for (auto rank = 0; rank < MPIEnviron_.sizeRank(); ++rank)
        {
            auto rankBuffer = buffer.getSend<valueType>(rank);
            for (auto data = 0; data < sendMap_[rank].size(); ++data)
                rankBuffer[data] = field(sendMap_[rank][data].local_idx);
        }
        buffer.startComm();
    }

    /**
     * @brief Checks if the non-blocking communication with the given name is complete.
     * @param commName The communication name, typically a file and line number.
     */
    void isComplete(std::string commName);

    /**
     * @brief Finalizes the non-blocking communication for a given field and communication name.
     * @tparam valueType The value type of the field.
     * @param field The field to be communicated/synchronised.
     * @param commName The communication name, typically a file and line number.
     */
    template<typename valueType>
    void finaliseComm(Field<valueType>& field, std::string commName)
    {
        NF_DEBUG_ASSERT(
            CommBuffer_.find(commName) != CommBuffer_.end(),
            "No communication buffer associated with key: " << commName
        );

        CommBuffer_[commName].waitComplete();
        for (auto rank = 0; rank < MPIEnviron_.sizeRank(); ++rank)
        {
            auto rankBuffer = CommBuffer_[commName].getReceive<valueType>(rank);
            for (auto data = 0; data < receiveMap_[rank].size(); ++data)
                field(receiveMap_[rank][data].local_idx) = rankBuffer[data];
        }
        CommBuffer_[commName].finaliseComm();
    }

private:

    mpi::MPIEnvironment MPIEnviron_; /**< The MPI environment. */
    RankSimplexCommMap sendMap_;     /**< The rank send map. */
    RankSimplexCommMap receiveMap_;  /**< The rank receive map. */
    std::unordered_map<std::string, mpi::FullDuplexCommBuffer>
        CommBuffer_; /**< The communication buffers. */

    /**
     * @brief Finds an uninitialized communication buffer.
     * @return An iterator to the uninitialized communication buffer, or end iterator if not found.
     */
    std::unordered_map<std::string, mpi::FullDuplexCommBuffer>::iterator findDuplexBuffer();

    /**
     * @brief Creates a new communication buffer with the given name.
     * @param commName The communication name.
     * @return An iterator to the newly created communication buffer.
     */
    std::unordered_map<std::string, mpi::FullDuplexCommBuffer>::iterator
    createNewDuplexBuffer(std::string commName);
};

} // namespace NeoFOAM
