// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include "NeoFOAM/core/executor/executor.hpp"
#include <type_traits>

namespace NeoFOAM
{

template<typename executor, typename Kernel>
void parallelForImpl(
    const executor& exec, label start, label end, Kernel kernel
)
{
    if constexpr (std::is_same<std::remove_reference_t<executor>, CPUExecutor>::
                      value)
    {
        for (label i = start; i < end; i++)
        {
            kernel(i);
        }
    }
    else
    {
        using runOn = typename executor::exec;
        Kokkos::parallel_for(
            "parallelForImpl", Kokkos::RangePolicy<runOn>(start, end), kernel
        );
    }
}


template<typename Kernel>
void parallelFor(NeoFOAM::Executor& exec, label start, label end, Kernel kernel)
{
    std::visit(
        [&](const auto& e) { parallelForImpl(e, start, end, kernel); }, exec
    );
};


template<typename executor, typename Kernel, typename T>
void parallelReduceImpl(
    const executor& exec, label start, label end, Kernel kernel, T& value
)
{
    if constexpr (std::is_same<std::remove_reference_t<executor>, CPUExecutor>::
                      value)
    {
        for (label i = start; i < end; i++)
        {
            kernel(i, value);
        }
    }
    else
    {
        using runOn = typename executor::exec;
        Kokkos::parallel_reduce(
            "parallelReduceImpl",
            Kokkos::RangePolicy<runOn>(start, end),
            kernel,
            value
        );
    }
}

template<typename Kernel, typename T>
void parallelReduce(
    NeoFOAM::Executor& exec, label start, label end, Kernel kernel, T& value
)
{
    return std::visit(
        [&](const auto& e)
        { return parallelReduceImpl(e, start, end, kernel, value); },
        exec
    );
};


} // namespace NeoFOAM
