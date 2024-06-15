// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <vector>
#include <string>
#include <span>

#include "NeoFOAM/core/mpi/environment.hpp"
#include "NeoFOAM/core/mpi/halfDuplexBuffer.hpp"

namespace NeoFOAM
{

namespace mpi
{


struct FullDuplexBuffer
{
    HalfDuplexBuffer send_;
    HalfDuplexBuffer receive_;

    FullDuplexBuffer() = default;
    FullDuplexBuffer(
        MPIEnvironment environ,
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
