// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <span>
#include <string>
#include <typeindex>
#include <vector>

#include <mpi.h>

#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/mpi/environment.hpp"
#include "NeoFOAM/core/mpi/operators.hpp"

namespace NeoFOAM
{

namespace mpi
{


    int bufferHash(const std::string& str) { return static_cast<int>(std::hash<std::string>(str)); }

    /**
     * @class HalfDuplexBuffer
     * @brief A class that represents a data buffer for communication in a distributed system.
     *
     * The Buffer class provides functionality for managing the data buffer for communication
     * between ranks in memory efficient and type flexible way. To prevent multiple reallocations,
     * the buffer size is never sized down.
     *
     * The maximum buffer size is determined by:
     * size = sum_r sum_cn sizeof(valueType)
     */
    class HalfDuplexBuffer
    {

    public:

        /**
         * @brief Default constructor
         */
        HalfDuplexBuffer() = default;

        /**
         * @brief Default destructor
         */
        ~HalfDuplexBuffer() = default;

        /**
         * @brief Construct a new Half Duplex Buffer object
         *
         * @param mpiEnviron The MPI environment.
         * @param rankCommSize The number of nodes per rank to be communicated with.
         */
        HalfDuplexBuffer(MPIEnvironment mpiEnviron, std::vector<std::size_t> rankCommSize)
            : mpiEnviron_(mpiEnviron)
        {
            setCommRankSize<char>(rankCommSize);
        }

        /**
         * @brief Set the MPI environment.
         *
         * @param mpiEnviron The MPI environment.
         */
        void setMPIEnvironment(MPIEnvironment mpiEnviron) { mpiEnviron_ = mpiEnviron; }

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
                !isCommInit(), "Communication buffer was initialised by name: " << commName_ << " ."
            );
            dataType = typeid(valueType);
            rankOffset_.resize(rankCommSize.size() + 1);
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
                !isCommInit(), "Communication buffer was initialised by name: " << commName_ << " ."
            );
            Buffer_.setType<valueType>();
            commName_ = commName;
            tag_ = bufferHash(key);
        }

        /**
         * @brief Set the data type for the buffer.
         */
        template<typename valueType>
        void setType()
        {
            NF_DEBUG_ASSERT(
                !isCommInit(), "Communication buffer was initialised by name: " << commName_ << " ."
            );
            if (0 == (sizeof(dataType) - sizeof(valueType))) return;
            updateDataSize(
                [&](const int rank)
                { return (rankOffset_[rank + 1] - rankOffset_[rank]) / sizeof(dataType); },
                sizeof(valueType)
            );
            dataType = typeid(valueType);
        }

        /**
         * @brief Get the communication name.
         *
         * @return const std::string& The name of the communication.
         */
        inline const std::string& getCommName() const { return commName_; }


        // actual communication

        /**
         * @brief Check if the communication is complete.
         * @return true if the communication is complete else false.
         */
        bool isComplete()
        {
            NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
            int flag;
            for (auto& request_ : request_)
            {
                int err = MPI_Test(&request_, &flag, MPI_STATUS_IGNORE);
                NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Test failed.");
                if (!flag) return false;
            }
            return static_cast<bool>(flag);
        }

        /**
         * @brief Post send for data to begin sending to all ranks this rank communicates with.
         */
        void send()
        {
            NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
            NF_DEBUG_ASSERT(isComplete(), "Communication buffer is already active.");
            for (auto rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
            {
                if (rank == mpiEnviron_.rank()) continue;
                sendScalar<char>(
                    rankBuffer_.data(),
                    rankBuffer_.size(),
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
            NF_DEBUG_ASSERT(isComplete(), "Communication buffer is already active.");
            for (auto rank = 0; rank < mpiEnviron_.sizeRank(); ++rank)
            {
                if (rank == mpiEnviron_.rank()) continue;
                recvScalar<char>(
                    rankBuffer_.data(),
                    rankBuffer_.size(),
                    rank,
                    tag_,
                    mpiEnviron_.comm(),
                    &request_[rank]
                );
            }
        }

        /**
         * @brief Finish the communication.
         */
        void waitComplete()
        {
            NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
            while (!isComplete())
            {
                // todo deadlock prevention.
                // wait for the communication to finish.
            }
        }


        void finaliseComm()
        {
            NF_DEBUG_ASSERT(isCommInit(), "Communication buffer is not initialised.");
            NF_DEBUG_ASSERT(isComplete(), "Cannot finalise while buffer is active.");
            tag_ = -1;
            commName_ = "unassigned";
            for (auto& request_ : request_)
                MPI_Request_free(&request_);
        }

        // data access

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
            NF_DEBUG_ASSERT(dataType == typeid(valueType), "Data type mismatch.");
            return std::span<valueType>(
                reinterpret_cast<valueType*>(rankBuffer_.data() + rankOffset_[rank]),
                (rankOffset_[rank + 1] - rankOffset_[rank]) / sizeof(valueType)
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
            NF_DEBUG_ASSERT(dataType == typeid(valueType), "Data type mismatch.");
            return std::span<const valueType>(
                reinterpret_cast<const valueType*>(rankBuffer_.data() + rankOffset_[rank]),
                (rankOffset_[rank + 1] - rankOffset_[rank]) / sizeof(valueType)
            );
        }

    private:

        int tag_ {-1};
        std::string commName_ {"unassigned"};
        std::type_index dataType {
            typeid(char)};                    /*< The data type currently stored in the buffer. */
        std::vector<char> rankBuffer_;        /*< The buffer data for all ranks. Never shrinks. */
        std::vector<std::size_t> rankOffset_; /*< The offset for a rank data in the buffer. */
        std::vector<MPI_Request> request_;
        MPIEnvironment mpiEnviron_;

        template<typename func>
        void updateDataSize(func rank_size, std::size_t data_size)
        {
            std::size_t dataSize = 0;
            for (auto rank = 0; rank < (rankOffset_.size() - 1); ++rank)
            {
                rankOffset_[rank] = dataSize;
                dataSize += rank_size(rank) * data_size;
            }
            rankOffset_.back() = dataSize;
            if (rankBuffer_.size() < dataSize) rankBuffer_.resize(dataSize); // we never size down.
        }
    };


    struct DuplexCommBuffer
    {
        HalfDuplexBuffer send_;
        HalfDuplexBuffer receive_;

        DuplexCommBuffer() = default;
        DuplexCommBuffer(
            mpi::MPIEnvironment environ,
            std::vector<std::size_t> sendSize,
            std::vector<std::size_t> receiveSize
        )
            : send_(environ, sendSize), receive_(environ, receiveSize) {};

        inline bool isCommInit() const { return send_.isCommInit() && receive_.isCommInit(); }

        /**
         * @brief Initialize the communication buffer.
         *
         * @tparam valueType The type of the data to be stored in the buffer.
         * @param commName A name for the communication, typically a file and line number.
         */
        template<typename valueType>
        void initComm(std::string commName)
        {
            send_.initComm<valueType>(commName);
            receive_.initComm<valueType>(commName);
        }

        template<typename valueType>
        std::span<valueType> get(const int rank)
        {
            return send_.(rank);
        }

        template<typename valueType>
        std::span<const valueType> get(const int rank) const
        {
            return receive_.(rank);
        }

        void startComm()
        {
            send_.send();
            receive_.receive();
        }

        bool isComplete() { return send_.isComplete() && receive_.isComplete(); }

        void waitComplete()
        {
            send_.waitComplete();
            receive_.waitComplete();
        }

        void finaliseComm()
        {
            send_.finaliseComm();
            receive_.finaliseComm();
        }
    };

}

}
