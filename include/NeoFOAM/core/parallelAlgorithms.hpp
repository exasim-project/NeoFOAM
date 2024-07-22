// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "NeoFOAM/core/types.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{


template<StorageType T>
class Field;


// Concept to check if a callable is compatible with void(const size_t)
template<typename Kernel>
concept parallelForKernel = requires(Kernel t, size_t i) {
    {
        t(i)
    } -> std::same_as<void>;
};

template<typename Executor, parallelForKernel Kernel>
void parallelFor(
    [[maybe_unused]] const Executor& exec, std::pair<size_t, size_t> range, Kernel kernel
)
{
    auto [start, end] = range;
    if constexpr (std::is_same<std::remove_reference_t<Executor>, CPUExecutor>::value)
    {
        for (size_t i = start; i < end; i++)
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
            KOKKOS_LAMBDA(const size_t i) { kernel(i); }
        );
    }
}


template<parallelForKernel Kernel>
void parallelFor(const NeoFOAM::Executor& exec, std::pair<size_t, size_t> range, Kernel kernel)
{
    std::visit([&](const auto& e) { parallelFor(e, range, kernel); }, exec);
}

// Concept to check if a callable is compatible with ValueType(const size_t)
template<typename Kernel, typename ValueType>
concept parallelForFieldKernel = requires(Kernel t, ValueType val, size_t i) {
    {
        t(i)
    } -> std::same_as<ValueType>;
};

template<typename Executor, StorageType T, parallelForFieldKernel<T> Kernel>
void parallelFor([[maybe_unused]] const Executor& exec, Field<T>& field, Kernel kernel)
{
    auto span = field.span();
    if constexpr (std::is_same<std::remove_reference_t<Executor>, CPUExecutor>::value)
    {
        for (size_t i = 0; i < field.size(); i++)
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
            KOKKOS_LAMBDA(const size_t i) { span[i] = kernel(i); }
        );
    }
}

template<StorageType T, parallelForFieldKernel<T> Kernel>
void parallelFor(Field<T>& field, Kernel kernel)
{
    std::visit([&](const auto& e) { parallelFor(e, field, kernel); }, field.exec());
}


template<typename Executor, typename Kernel, typename T>
void parallelReduce(
    [[maybe_unused]] const Executor& exec, std::pair<size_t, size_t> range, Kernel kernel, T& value
)
{
    auto [start, end] = range;
    if constexpr (std::is_same<std::remove_reference_t<Executor>, CPUExecutor>::value)
    {
        for (size_t i = start; i < end; i++)
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
void parallelReduce(
    const NeoFOAM::Executor& exec, std::pair<size_t, size_t> range, Kernel kernel, T& value
)
{
    return std::visit([&](const auto& e) { return parallelReduce(e, range, kernel, value); }, exec);
}


template<typename Executor, StorageType T, typename Kernel, typename U>
void parallelReduce([[maybe_unused]] const Executor& exec, Field<T>& field, Kernel kernel, U& value)
    requires std::convertible_to<U, T>
{
    if constexpr (std::is_same<std::remove_reference_t<Executor>, CPUExecutor>::value)
    {
        for (size_t i = 0; i < field.size(); i++)
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

template<StorageType T, typename Kernel, typename U>
void parallelReduce(Field<T>& field, Kernel kernel, U& value)
    requires std::convertible_to<U, T>
{
    return std::visit(
        [&](const auto& e) { return parallelReduce(e, field, kernel, value); }, field.exec()
    );
}


} // namespace NeoFOAM
