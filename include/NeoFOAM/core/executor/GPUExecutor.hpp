// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>

namespace NeoFOAM
{

/**
 * @class GPUExecutor
 * @brief Executor for GPU offloading.
 *
 * @ingroup Executor
 */
class GPUExecutor
{
public:

    using exec = Kokkos::DefaultExecutionSpace;

    GPUExecutor();
    ~GPUExecutor();

    template<typename T>
    T* alloc(size_t size) const
    {
        return static_cast<T*>(Kokkos::kokkos_malloc<exec>("Field", size * sizeof(T)));
    }

    template<typename T>
    T* realloc(void* ptr, size_t newSize) const
    {
        return static_cast<T*>(Kokkos::kokkos_realloc<exec>(ptr, newSize * sizeof(T)));
    }

    template<typename ValueType>
    decltype(auto) createKokkosView(ValueType* ptr, size_t size) const
    {
        return Kokkos::View<ValueType*, Kokkos::DefaultExecutionSpace, Kokkos::MemoryUnmanaged>(
            ptr, size
        );
    }

    void* alloc(size_t size) const { return Kokkos::kokkos_malloc<exec>("Field", size); }

    void* realloc(void* ptr, size_t newSize) const
    {
        return Kokkos::kokkos_realloc<exec>(ptr, newSize);
    }

    std::string print() const { return std::string(exec::name()); }

    void free(void* ptr) const noexcept { Kokkos::kokkos_free<exec>(ptr); }

    std::string name() const { return "GPUExecutor"; };
};

} // namespace NeoFOAM
