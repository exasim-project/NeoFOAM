// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 NeoFOAM authors

#include "NeoFOAM/linearAlgebra/utilities.hpp"


namespace NeoFOAM::la
{


// TODO: check if this can be replaced by Ginkgos executor mapping
#if NF_WITH_GINKGO
std::shared_ptr<gko::Executor> getGkoExecutor(Executor exec)
{
    return std::visit(
        [](auto concreteExec) -> std::shared_ptr<gko::Executor>
        {
            using ExecType = std::decay_t<decltype(concreteExec)>;
            if constexpr (std::is_same_v<ExecType, SerialExecutor>)
            {
                return gko::ReferenceExecutor::create();
            }
            else if constexpr (std::is_same_v<ExecType, CPUExecutor>)
            {
#if defined(KOKKOS_ENABLE_OMP)
                return gko::OmpExecutor::create();
#elif defined(KOKKOS_ENABLE_THREADS)
                return gko::ReferenceExecutor::create();
#endif
            }
            else if constexpr (std::is_same_v<ExecType, GPUExecutor>)
            {
#if defined(KOKKOS_ENABLE_CUDA)
                return gko::CudaExecutor::create(
                    Kokkos::device_id(),
                    gko::ReferenceExecutor::create(),
                    std::make_shared<gko::CudaAllocator>()
                    // TODO: check if this is correct
                    // concreteExec.cuda_stream()
                );
#elif defined(KOKKOS_ENABLE_HIP)
                return gko::HipExecutor::create(
                    Kokkos::device_id(),
                    gko::ReferenceExecutor::create(),
                    std::make_shared<gko::HipAllocator>()
                    // TODO: check if this is correct
                    // concreteExec.hip_stream()
                );
#endif
                throw std::runtime_error("No valid GPU executor mapping available");
            }
            else
            {
                throw std::runtime_error("Unsupported executor type");
            }
        },
        exec
    );
}

#endif
}
