// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <span>
#include <string>
#include <variant>
#include <vector>

#include "NeoFOAM/core/mpi/environment.hpp"
#include "NeoFOAM/core/mpi/halfDuplexCommBuffer.hpp"

namespace NeoFOAM
{

namespace mpi
{

/**
 * @class FullDuplexCommBuffer
 * @brief A buffer for full-duplex communication in a distributed system using MPI.
 * @tparam MemorySpace The memory space to use for the communication buffer.
 *
 * The FullDuplexCommBuffer class facilitates efficient, non-blocking, point-to-point data
 * exchange between MPI ranks, allowing for simultaneous send and receive operations. It
 * manages two HalfDuplexCommBuffer instances: one for sending data and one for receiving data.
 */
template<class MemorySpace = Kokkos::HostSpace>
class FullDuplexCommBuffer
{
public:

    /**
     * @brief Default constructor.
     */
    FullDuplexCommBuffer() = default;

    /**
     * @brief Constructor that initializes the send and receive buffers.
     * @param environ The MPI environment.
     * @param sendSize The number of nodes, per rank, that this rank sends to.
     * @param receiveSize The number of nodes, per rank, that this rank receives from.
     */
    FullDuplexCommBuffer(
        MPIEnvironment mpiEnviron,
        std::vector<std::size_t> sendSize,
        std::vector<std::size_t> receiveSize
    )
        : send_(mpiEnviron, sendSize), receive_(mpiEnviron, receiveSize) {};

    /**
     * @brief Check if the communication buffers are initialized.
     * @return True if the buffers are initialized, false otherwise.
     */
    inline bool isCommInit() const { return send_.isCommInit() && receive_.isCommInit(); }

    /**
     * @brief Initialize the communication buffer.
     * @tparam valueType The type of the data to be stored in the buffer.
     * @param commName A name for the communication, typically a file and line number.
     */
    template<typename valueType>
    void initComm(std::string commName)
    {
        send_.template initComm<valueType>(commName);
        receive_.template initComm<valueType>(commName);
    }

    /**
     * @brief Gets a span of data for the send buffer for a specific rank.
     * @tparam valueType The type of the data.
     * @param rank The rank of the send buffer to get.
     * @return A span of data for the send buffer.
     */
    template<typename valueType>
    std::span<valueType> getSend(const int rank)
    {
        return send_.template get<valueType>(rank);
    }

    /**
     * @brief Gets a span of data for the send buffer for a specific rank.
     * @tparam valueType The type of the data.
     * @param rank The rank of the send buffer to get.
     * @return A span of data for the send buffer.
     */
    template<typename valueType>
    std::span<const valueType> getSend(const int rank) const
    {
        return send_.template get<valueType>(rank);
    }

    /**
     * @brief Gets a span of data for the receive buffer for a specific rank.
     * @tparam valueType The type of the data.
     * @param rank The rank of the receive buffer to get.
     * @return A span of data for the receive buffer.
     */
    template<typename valueType>
    std::span<valueType> getReceive(const int rank)
    {
        return receive_.template get<valueType>(rank);
    }

    /**
     * @brief Gets a span of data for the receive buffer for a specific rank.
     * @tparam valueType The type of the data.
     * @param rank The rank of the receive buffer to get.
     * @return A span of data for the receive buffer.
     */
    template<typename valueType>
    std::span<const valueType> getReceive(const int rank) const
    {
        return receive_.template get<valueType>(rank);
    }

    /**
     * @brief Start non-blocking communication by sending and receiving data.
     */
    inline void startComm()
    {
        send_.send();
        receive_.receive();
    }

    /**
     * @brief Check if the communication is complete.
     * @return True if the communication is complete, false otherwise.
     */
    inline bool isComplete() { return send_.isActive() && receive_.isActive(); }

    /**
     * @brief Blocking wait for the communication to complete.
     */
    inline void waitComplete()
    {
        send_.waitComplete();
        receive_.waitComplete();
    }

    /**
     * @brief Finalize the communication by cleaning up the buffers.
     */
    inline void finaliseComm()
    {
        send_.finaliseComm();
        receive_.finaliseComm();
    }

private:

    HalfDuplexCommBuffer<MemorySpace> send_;    /**< The send buffer. */
    HalfDuplexCommBuffer<MemorySpace> receive_; /**< The receive buffer. */
};

// using BufferVariant = std::variant FullDuplexCommBuffer<Kokkos::HostSpace>,
// #ifdef KOKKOS_ENABLE_CUDA
//       FullDuplexCommBuffer<Kokkos::CudaSpace>, FullDuplexCommBuffer<Kokkos::CudaUVMSpace>,
// #endif
// #ifdef KOKKOS_ENABLE_HIP
//       FullDuplexCommBuffer<Kokkos::HIPSpace>,
// #endif
// #ifdef KOKKOS_ENABLE_OPENMPTARGET
//       FullDuplexCommBuffer<Kokkos::OpenMPTargetSpace>,
// #endif
// #ifdef KOKKOS_ENABLE_SYCL
//       FullDuplexCommBuffer<Kokkos::SYCLSpace>, FullDuplexCommBuffer<Kokkos::SYCLSharedUSMSpace>,
//       FullDuplexCommBuffer<Kokkos::SYCLDeviceUSMSpace>,
// #endif
//       FullDuplexCommBuffer<Kokkos::AnonymousSpace> > ;

}

}
