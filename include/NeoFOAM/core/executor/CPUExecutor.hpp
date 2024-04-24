// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>

namespace NeoFOAM
{

/**
 * @class CPUExecutor
 * @brief Reference executor for serial CPU execution.
 *
 * @ingroup Executor
 */
class CPUExecutor
{
public:

    using exec = Kokkos::Serial;

    CPUExecutor();
    ~CPUExecutor();

    template<typename T>
    T* alloc(size_t size) const
    {
        return static_cast<T*>(
            Kokkos::kokkos_malloc<exec>("Field", size * sizeof(T))
        );
    }

    template<typename T>
    T* realloc(void* ptr, size_t new_size) const
    {
        return static_cast<T*>(
            Kokkos::kokkos_realloc<exec>(ptr, new_size * sizeof(T))
        );
    }

    void* alloc(size_t size) const
    {
        return Kokkos::kokkos_malloc<exec>("Field", size);
    }

    void* realloc(void* ptr, size_t new_size) const
    {
        return Kokkos::kokkos_realloc<exec>(ptr, new_size);
    }

    std::string print() const { return std::string(exec::name()); }

    void free(void* ptr) const noexcept
    {
        Kokkos::kokkos_free<exec>(ptr);
    };
};

} // namespace NeoFOAM
