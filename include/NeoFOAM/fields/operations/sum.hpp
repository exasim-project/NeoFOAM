// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{

struct SumKernel
{
    template<typename T>
    void operator()(const GPUExecutor& exec, const Field<T>& field, T& sum) const
    {
        using executor = typename GPUExecutor::exec;
        auto field_f = field.field();
        auto end = field.size();
        Kokkos::parallel_reduce(
            "sum", Kokkos::RangePolicy<executor>(0, end), KOKKOS_LAMBDA(const int i, T& lsum) {
                lsum += field_f[i];
            },
            sum
        );
    }

    template<typename T>
    void operator()(const OMPExecutor& exec, const Field<T>& field, T& sum)
    {
        using executor = typename OMPExecutor::exec;
        auto field_f = field.field();
        auto end = field.size();
        Kokkos::parallel_reduce(
            "sum", Kokkos::RangePolicy<executor>(0, end), KOKKOS_LAMBDA(const int i, T& lsum) {
                lsum += field_f[i];
            },
            sum
        );
    }

    template<typename T>
    void operator()(const CPUExecutor& exec, const Field<T>& field, T& sum)
    {
        using executor = typename CPUExecutor::exec;
        auto field_f = field.field();
        auto end = field.size();
        Kokkos::parallel_reduce(
            "sum", Kokkos::RangePolicy<executor>(0, end), KOKKOS_LAMBDA(const int i, T& lsum) {
                lsum += field_f[i];
            },
            sum
        );
    }
};


template<typename T>
T sum(const Field<T>& field)
{
    T sumValue {};
    SumKernel kernel {};
    std::visit([&](const auto& exec)
               { kernel(exec, field, sumValue); },
               field.exec());
    return sumValue;
};

template<>
scalar sum(const Field<scalar>& field)
{
    scalar sumValue = 0;
    SumKernel kernel {};
    std::visit([&](const auto& exec)
               { kernel(exec, field, sumValue); },
               field.exec());
    return sumValue;
};


} // namespace NeoFOAM
