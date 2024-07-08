// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include "NeoFOAM/core/executor/executor.hpp"
#include <type_traits>

namespace NeoFOAM
{

template<typename Executor, typename Kernel>
void parallelFor(const Executor& exec, label start, label end, Kernel kernel)
{
    if constexpr (std::is_same<std::remove_reference_t<Executor>, CPUExecutor>::value)
    {
        for (label i = start; i < end; i++)
        {
            kernel(i);
        }
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_for(
            "parallelFor",
            Kokkos::RangePolicy<runOn>(start, end),
            KOKKOS_LAMBDA(const label i) { kernel(i); }
        );
    }
}


template<typename Kernel>
void parallelFor(NeoFOAM::Executor& exec, label start, label end, Kernel kernel)
{
    std::visit([&](const auto& e) { parallelFor(e, start, end, kernel); }, exec);
}

template<typename Executor, typename ValueType, typename Kernel>
void parallelFor(const Executor& exec, Field<ValueType>& field, Kernel kernel)
{
    auto span = field.span();
    if constexpr (std::is_same<std::remove_reference_t<Executor>, CPUExecutor>::value)
    {
        for (label i = 0; i < field.size(); i++)
        {
            span[i] = kernel(i);
        }
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_for(
            "parallelFor",
            Kokkos::RangePolicy<runOn>(0, field.size()),
            KOKKOS_LAMBDA(const label i) { span[i] = kernel(i); }
        );
    }
}

template<typename ValueType, typename Kernel>
void parallelFor(Field<ValueType>& field, Kernel kernel)
{
    std::visit([&](const auto& e) { parallelFor(e, field, kernel); }, field.exec());
}


template<typename Executor, typename Kernel, typename T>
void parallelReduce(const Executor& exec, label start, label end, Kernel kernel, T& value)
{
    if constexpr (std::is_same<std::remove_reference_t<Executor>, CPUExecutor>::value)
    {
        for (label i = start; i < end; i++)
        {
            kernel(i, value);
        }
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_reduce(
            "parallelReduce", Kokkos::RangePolicy<runOn>(start, end), kernel, value
        );
    }
}

template<typename Kernel, typename T>
void parallelReduce(NeoFOAM::Executor& exec, label start, label end, Kernel kernel, T& value)
{
    return std::visit(
        [&](const auto& e) { return parallelReduce(e, start, end, kernel, value); }, exec
    );
}


template<typename Executor, typename ValueType, typename Kernel, typename T>
void parallelReduce(const Executor& exec, Field<ValueType>& field, Kernel kernel, T& value)
{
    if constexpr (std::is_same<std::remove_reference_t<Executor>, CPUExecutor>::value)
    {
        for (label i = 0; i < field.size(); i++)
        {
            kernel(i, value);
        }
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_reduce(
            "parallelReduce", Kokkos::RangePolicy<runOn>(0, field.size()), kernel, value
        );
    }
}

template<typename ValueType, typename Kernel, typename T>
void parallelReduce(Field<ValueType>& field, Kernel kernel, T& value)
{
    return std::visit(
        [&](const auto& e) { return parallelReduce(e, field, kernel, value); }, field.exec()
    );
}


} // namespace NeoFOAM
