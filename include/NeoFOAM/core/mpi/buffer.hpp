// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <mpi.h>
#include <span>
#include <typeindex>
#include <vector>

#include "NeoFOAM/core/error.hpp"

namespace NeoFOAM
{

namespace mpi
{

    /**
     * @class Buffer
     * @brief A class that represents a data buffer for communication in a distributed system.
     *
     * The Buffer class provides functionality for managing the data buffer for communication
     * between ranks in memory efficient and type flexible way. To prevent multiple reallocations,
     * the buffer size is never sized down.
     *
     * The maximum buffer size is determined by:
     * size = sum_r sum_cn sizeof(valueType)
     */
    class Buffer
    {

    public:

        /**
         * @brief Default constructor
         */
        Buffer() = default;

        /**
         * @brief Default destructor
         */
        ~Buffer() = default;

        /**
         * @brief Set the buffer data size based on the rank communication and value type.
         *
         * @tparam valueType The type of the data to be stored in the buffer.
         * @param rankCommSize The number of nodes to be communicated between each rank.
         */
        template<typename valueType>
        void setCommTypeSize(std::vector<std::size_t> rankCommSize)
        {
            dataType = typeid(valueType);
            rankOffset.resize(rankCommSize.size() + 1);
            updateDataSize([&](const int rank) { return rankCommSize[rank]; }, sizeof(valueType));
        }

        /**
         * @brief TODO
         */
        template<typename valueType>
        void setCommType()
        {
            if (0 == (sizeof(dataType) - sizeof(valueType))) return;
            updateDataSize(
                [&](const int rank)
                { return (rankOffset[rank + 1] - rankOffset[rank]) / sizeof(dataType); },
                sizeof(valueType)
            );
            dataType = typeid(valueType);
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
                (rankOffset[rank + 1] - rankOffset[rank]) / sizeof(valueType)
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
                reinterpret_cast<const valueType*>(rankBuffer.data() + rankOffset[rank]),
                (rankOffset[rank + 1] - rankOffset[rank]) / sizeof(valueType)
            );
        }

    private:

        std::type_index dataType {
            typeid(char)};                   /*< The data type currently stored in the buffer. */
        std::vector<char> rankBuffer;        /*< The buffer data for all ranks. Never shrinks. */
        std::vector<std::size_t> rankOffset; /*< The offset for a rank data in the buffer. */


        template<typename func>
        void updateDataSize(func rank_size, std::size_t data_size)
        {
            std::size_t dataSize = 0;
            for (auto rank = 0; rank < (rankOffset.size() - 1); ++rank)
            {
                rankOffset[rank] = dataSize;
                dataSize += rank_size(rank) * data_size;
            }
            rankOffset.back() = dataSize;
            if (rankBuffer.size() < dataSize) rankBuffer.resize(dataSize); // we never size down.
        }
    };

}

}
