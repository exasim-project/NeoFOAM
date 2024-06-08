// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <complex>
#include <mpi.h>
#include <type_traits>

#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/primitives/vector.hpp"

namespace NeoFOAM
{

namespace mpi
{

    enum class ReduceOp
    {
        Max,
        Min,
        Sum,
        Prod,
        Land,
        Band,
        Lor,
        Bor,
        Maxloc,
        Minloc
    };

    constexpr MPI_Op getOp(const ReduceOp op)
    {
        switch (op)
        {
        case ReduceOp::Max:
            return MPI_MAX;
        case ReduceOp::Min:
            return MPI_MIN;
        case ReduceOp::Sum:
            return MPI_SUM;
        case ReduceOp::Prod:
            return MPI_PROD;
        case ReduceOp::Land:
            return MPI_LAND;
        case ReduceOp::Band:
            return MPI_BAND;
        case ReduceOp::Lor:
            return MPI_LOR;
        case ReduceOp::Bor:
            return MPI_BOR;
        case ReduceOp::Maxloc:
            return MPI_MAXLOC;
        case ReduceOp::Minloc:
            return MPI_MINLOC;
        default:
            NF_ERROR_EXIT("Invalid MPI reduce operation requested.");
            return MPI_LOR; // This is to suppress the warning
        }
    }


    template<typename valueType>
    constexpr MPI_Datatype
    getType() // maybe try to conseval this -> you can template but then compile errors are long!!!
    {
        if constexpr (std::is_same_v<valueType, char>) return MPI_CHAR;
        else if constexpr (std::is_same_v<valueType, short>)
            return MPI_SHORT;
        else if constexpr (std::is_same_v<valueType, int>)
            return MPI_INT;
        else if constexpr (std::is_same_v<valueType, long>)
            return MPI_LONG;
        else if constexpr (std::is_same_v<valueType, long long>)
            return MPI_LONG_LONG;
        else if constexpr (std::is_same_v<valueType, signed char>)
            return MPI_SIGNED_CHAR;
        else if constexpr (std::is_same_v<valueType, unsigned char>)
            return MPI_UNSIGNED_CHAR;
        else if constexpr (std::is_same_v<valueType, unsigned short>)
            return MPI_UNSIGNED_SHORT;
        else if constexpr (std::is_same_v<valueType, unsigned>)
            return MPI_UNSIGNED;
        else if constexpr (std::is_same_v<valueType, unsigned long>)
            return MPI_UNSIGNED_LONG;
        else if constexpr (std::is_same_v<valueType, unsigned long long>)
            return MPI_UNSIGNED_LONG_LONG;
        else if constexpr (std::is_same_v<valueType, float>)
            return MPI_FLOAT;
        else if constexpr (std::is_same_v<valueType, double>)
            return MPI_DOUBLE;
        else if constexpr (std::is_same_v<valueType, long double>)
            return MPI_LONG_DOUBLE;
        else if constexpr (std::is_same_v<valueType, wchar_t>)
            return MPI_WCHAR;
        else if constexpr (std::is_same_v<valueType, int8_t>)
            return MPI_INT8_T;
        else if constexpr (std::is_same_v<valueType, int16_t>)
            return MPI_INT16_T;
        else if constexpr (std::is_same_v<valueType, int32_t>)
            return MPI_INT32_T;
        else if constexpr (std::is_same_v<valueType, int64_t>)
            return MPI_INT64_T;
        else if constexpr (std::is_same_v<valueType, uint8_t>)
            return MPI_UINT8_T;
        else if constexpr (std::is_same_v<valueType, uint16_t>)
            return MPI_UINT16_T;
        else if constexpr (std::is_same_v<valueType, uint32_t>)
            return MPI_UINT32_T;
        else if constexpr (std::is_same_v<valueType, uint64_t>)
            return MPI_UINT64_T;
        else if constexpr (std::is_same_v<valueType, bool>)
            return MPI_CXX_BOOL;
        else if constexpr (std::is_same_v<valueType, std::complex<float>>)
            return MPI_CXX_FLOAT_COMPLEX;
        else if constexpr (std::is_same_v<valueType, std::complex<double>>)
            return MPI_CXX_DOUBLE_COMPLEX;
        else if constexpr (std::is_same_v<valueType, std::complex<long double>>)
            return MPI_CXX_LONG_DOUBLE_COMPLEX;
        else
            NF_ERROR_EXIT("Invalid MPI datatype requested.");
    }

    template<typename valueType>
    void reduceAllScalar(valueType* value, const ReduceOp op, MPI_Comm comm)
    {
        MPI_Allreduce(value, value, 1, getType<valueType>(), getOp(op), comm);
    }

    template<typename valueType>
    void sendScalar(
        const valueType* buffer,
        const int size,
        int r_rank,
        int tag,
        MPI_Comm comm,
        MPI_Request* request
    )
    {
        int err = MPI_Isend(buffer, size, getType<valueType>(), r_rank, tag, comm, *request);
        NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Isend failed.");
    }

    template<typename valueType>
    void recvScalar(
        valueType* buffer, const int size, int s_rank, int tag, MPI_Comm comm, MPI_Request* request
    )
    {
        int err = MPI_Irecv(buffer, size, getType<valueType>(), s_rank, tag, comm, *request);
        NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Irecv failed.");
    }

    bool test(MPI_Request* request)
    {
        int flag;
        int err = MPI_Test(request, &flag, MPI_STATUS_IGNORE);
        NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Test failed.");
        return static_cast<bool>(flag);
    }

} // namespace mpi

}
