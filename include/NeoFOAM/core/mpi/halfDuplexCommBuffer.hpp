// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <span>
#include <string>
#include <typeindex>
#include <vector>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/mpi/environment.hpp"
#include "NeoFOAM/core/mpi/operators.hpp"

namespace NeoFOAM
{

namespace mpi
{

/**
 * @brief A hash function for a string.
 *
 * @param str The string to be hashed.
 * @return int The hash value of the string.
 */
inline int bufferHash(const std::string& str)
{
    // There is also an MPI environment value for that, but somehow it doesn't work using that
    // Also reserve 10 tags for other uses
    constexpr int maxTagValue = 32767 - 10;
    std::size_t tag = std::hash<std::string> {}(str);
    tag &= 0x7FFFFFFF; // turn 'int' signed bit to 0, MPI does not like negative tags.
    return (static_cast<int>(tag) % maxTagValue) + 10;
}


template<typename Type>
concept MemorySpace = requires {
    typename Type::execution_space;
    {
        Type::is_memory_space
    } -> std::convertible_to<bool>;
    requires Type::is_memory_space == true;
};

/**
 * @class HalfDuplexCommBuffer
 * @brief A data buffer for half-duplex communication in a distributed system using MPI.
 *
 * The HalfDuplexCommBuffer class facilitates efficient, non-blocking, point-to-point data exchange
 * between MPI ranks in a distributed system. It maintains a buffer used for data communication,
 * capable of handling various data types. The buffer does not shrink once initialized, minimizing
 * memory reallocation and improving memory efficiency. The class operates in a half-duplex mode,
 * meaning it is either sending or receiving data at any given time.
 *
 * States and changes of states:
 * 1. Initialized: Is the buffer initialized for a communication, with a name and data type.
 * 2. Active: The buffer is actively sending or receiving data.
 *
 * These states can be queried using the isCommInit() and isActive() functions. However isActive()
 * can only be called when the buffer is initialized, where as isCommInit() can be called at any
 * time.
 *
 * The states are changed through the following functions:
 * 1. initComm(): Sets the buffer to an initialized state.
 * 2. send() & receive(): Sets the buffer to active state.
 * 3. waitComplete(): One return sets the buffer to an inactive state.
 * 4. finaliseComm(): Sets the buffer to an uninitialized state.
 *
 * It is critical once the data has been copied out of the buffer, the buffer set back to an
 * uninitialized state so it can be re-used.
 */
template<class MemorySpace = Kokkos::HostSpace>
class HalfDuplexCommBuffer
{

public:

    /**
     * @brief Default constructor
     */
    HalfDuplexCommBuffer() = default;

    /**
     * @brief Default destructor
     */
    ~HalfDuplexCommBuffer() = default;

    /**
     * @brief Construct a new Half Duplex Buffer object
     *
     * @param mpiEnviron The MPI environment.
     * @param rankCommSize The number of nodes per rank to be communicated with.
     */
    HalfDuplexCommBuffer(MPIEnvironment mpiEnviron, std::vector<std::size_t> rankCommSize)
        : mpiEnviron_(mpiEnviron)
    {
        setCommRankSize<char>(rankCommSize);
    }

    /**
     * @brief Set the MPI environment.
     *
     * @param mpiEnviron The MPI environment.
     */
    inline void setMPIEnvironment(MPIEnvironment mpiEnviron) { mpiEnviron_ = mpiEnviron; }

    /**
     * @brief Get the communication name.
     *
     * @return const std::string& The name of the communication.
     */
    inline bool isCommInit() const { return tag_ != -1; }

    /**
     * @brief Set the buffer data size based on the rank communication and value type.
     *
     * @tparam valueType The type of the data to be stored in the buffer.
     * @param rankCommSize The number of nodes per rank to be communicated with.
     */
    template<typename valueType>
    void setCommRankSize(std::vector<std::size_t> rankCommSize)
    {
        NF_DEBUG_ASSERT(
            !isCommInit(), "Communication buffer was initialised by name: " << commName_ << "."
        );
        NF_DEBUG_ASSERT(
            rankCommSize.size() == mpiEnviron_.sizeRank(),
            "Rank size mismatch. " << rankCommSize.size() << " vs. " << mpiEnviron_.sizeRank()
        );
        typeSize_ = sizeof(valueType);
        Kokkos::resize(rankOffsetSpace_, rankCommSize.size() + 1);
        Kokkos::resize(rankOffsetHost_, rankCommSize.size() + 1);
        request_.resize(rankCommSize.size(), MPI_REQUEST_NULL);
        updateDataSize([&](const int rank) { return rankCommSize[rank]; }, sizeof(valueType));
    }

    /**
     * @brief Initialize the communication buffer.
     *
     * @tparam valueType The type of the data to be stored in the buffer.
     * @param commName A name for the communication, typically a file and line number.
     */
    template<typename valueType>
    void initComm(std::string commName)
    {
        NF_DEBUG_ASSERT(
            !isCommInit(), "Communication buffer was initialised by name: " << commName_ << "."
        );
        setType<valueType>();
        commName_ = commName;
        tag_ = bufferHash(commName);
    }

    /**
     * @brief Get the communication name.
     *
     * @return const std::string& The name of the communication.
     */
    inline const std::string& getCommName() const { return commName_; }

    /**
     * @brief Check if the active communication is complete.
     * @return true if the communication is complete else false.
     */
    bool isActive()
    {
        NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
        bool isActive = false;
        for (auto& request : request_)
        {
            if (request != MPI_REQUEST_NULL)
            {
                int flag = 0;
                int err = MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
                NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Test failed.");
                if (!flag)
                {
                    isActive = true;
                    break;
                }
            }
        }
        return isActive;
    }

    /**
     * @brief Post send for data to begin sending to all ranks this rank communicates with.
     */
    void send()
    {
        NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
        NF_DEBUG_ASSERT(!isActive(), "Communication buffer is already actively sending.");
        for (auto rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
        {
            if (rankOffsetHost_(rank + 1) - rankOffsetHost_(rank) == 0) continue;
            isend<char>(
                rankBuffer_.data() + rankOffsetHost_(rank),
                rankOffsetHost_(rank + 1) - rankOffsetHost_(rank),
                rank,
                tag_,
                mpiEnviron_.comm(),
                &request_[rank]
            );
        }
    }

    /**
     * @brief Post receive for data to begin receiving from all ranks this rank communicates
     * with.
     */
    void receive()
    {
        NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
        NF_DEBUG_ASSERT(!isActive(), "Communication buffer is already actively receiving.");
        for (auto rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
        {
            if (rankOffsetHost_(rank + 1) - rankOffsetHost_(rank) == 0) continue;
            irecv<char>(
                rankBuffer_.data() + rankOffsetHost_(rank),
                rankOffsetHost_(rank + 1) - rankOffsetHost_(rank),
                rank,
                tag_,
                mpiEnviron_.comm(),
                &request_[rank]
            );
        }
    }


    /**
     * @brief Blocking wait for the communication to finish.
     */
    void waitComplete()
    {
        NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
        while (isActive())
        {
            // todo deadlock prevention.
            // wait for the communication to finish.
        }
    }

    /**
     * @brief Finalise the communication.
     */
    void finaliseComm()
    {
        NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
        NF_DEBUG_ASSERT(!isActive(), "Cannot finalise while buffer is active.");
        for (auto& request : request_)
            NF_DEBUG_ASSERT(
                request == MPI_REQUEST_NULL, "MPI_Request not null, communication not complete."
            );
        tag_ = -1;
        commName_ = "unassigned";
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
        NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
        NF_DEBUG_ASSERT(typeSize_ == sizeof(valueType), "Data type (size) mismatch.");
        return std::span<valueType>(
            reinterpret_cast<valueType*>(rankBuffer_.data() + rankOffsetSpace_(rank)),
            (rankOffsetSpace_(rank + 1) - rankOffsetSpace_(rank)) / sizeof(valueType)
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
        NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
        NF_DEBUG_ASSERT(typeSize_ == sizeof(valueType), "Data type (size) mismatch.");
        return std::span<const valueType>(
            reinterpret_cast<const valueType*>(rankBuffer_.data() + rankOffsetSpace_(rank)),
            (rankOffsetSpace_(rank + 1) - rankOffsetSpace_(rank)) / sizeof(valueType)
        );
    }

private:

    int tag_ {-1};                        /*< The tag for the communication. */
    std::string commName_ {"unassigned"}; /*< The name of the communication. */
    std::size_t typeSize_ {sizeof(char)}; /*< The data type currently stored in the buffer. */
    MPIEnvironment mpiEnviron_;           /*< The MPI environment. */
    std::vector<MPI_Request> request_;    /*< The MPI request for communication with each rank. */
    Kokkos::View<char*, MemorySpace>
        rankBuffer_; /*< The buffer data for all ranks. Never shrinks. */
    Kokkos::View<std::size_t*, MemorySpace>
        rankOffsetSpace_; /*< The offset (in bytes) for a rank data in the buffer. */
    Kokkos::View<std::size_t*, Kokkos::HostSpace>
        rankOffsetHost_; /*< The offset (in bytes) for a rank data used for MPI communication. */

    /**
     * @brief Set the data type for the buffer.
     */
    template<typename valueType>
    void setType()
    {
        NF_DEBUG_ASSERT(
            !isCommInit(), "Communication buffer was initialised by name: " << commName_ << "."
        );
        if (0 == (typeSize_ - sizeof(valueType))) return;
        updateDataSize(
            [rankOffset = rankOffsetSpace_, typeSize = typeSize_](const int rank)
            { return (rankOffset(rank + 1) - rankOffset(rank)) / typeSize; },
            sizeof(valueType)
        );

        typeSize_ = sizeof(valueType);
    }

    /**
     * @brief Update the buffer data size using a lambda (for the size per rank) and type size.
     *
     * @tparam func The lambda type for the rank size.
     * @param rankSize The function to get the rank size.
     * @param dataSize The size of the data to be stored in the buffer.
     */
    template<typename func>
    void updateDataSize(func rankSize, std::size_t newSize)
    {
        std::size_t dataSize = 0;

        // This works because rankOffsetHost_ is guaranteed to be on host.
        for (auto rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
        {
            rankOffsetHost_(rank) = dataSize;
            dataSize += rankSize(rank) * newSize;
        }
        rankOffsetHost_(mpiEnviron_.sizeRank()) = dataSize;

        if (rankBuffer_.size() < dataSize) Kokkos::resize(rankBuffer_, dataSize);
        Kokkos::deep_copy(rankOffsetSpace_, rankOffsetHost_);
    }
};

}

}
