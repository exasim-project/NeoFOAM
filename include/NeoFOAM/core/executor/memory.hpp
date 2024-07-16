// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once


#include <variant>
#include <Kokkos_Core.hpp>

namespace NeoFOAM
{
using kokkosMemory = std::variant<
    Kokkos::HostSpace,
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::CudaSpace,
    Kokkos::CudaUVMSpace,
    Kokkos::CudaHostPinnedSpace,
#endif
#ifdef KOKKOS_ENABLE_HIP
    Kokkos::HIPSpace,
    Kokkos::HIPHostPinnedSpace,
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    Kokkos::OpenMPTargetSpace,
#endif
#ifdef KOKKOS_ENABLE_SYCL
    Kokkos::SYCLSpace,
    Kokkos::SYCLSharedUSMSpace,
    Kokkos::SYCLDeviceUSMSpace,
#endif
#ifdef KOKKOS_ENABLE_HPX
    Kokkos::HPXSpace,
#endif
    Kokkos::AnonymousSpace>;

void* alloc(kokkosMemory& memory_space, size_t size)
{
    return std::visit(
        [size](auto&& space)
        {
            using MemorySpace = std::decay_t<decltype(space)>;
            return Kokkos::kokkos_malloc<MemorySpace>(size);
        },
        memory_space
    );
}

}
