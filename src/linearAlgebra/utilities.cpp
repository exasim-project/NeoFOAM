// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 NeoFOAM authors

#include "NeoFOAM/linearAlgebra/utilities.hpp"


namespace NeoFOAM::la
{


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
                return gko::OmpExecutor::create();
            }
            // else if constexpr (std::is_same_v<ExecType, GPUExecutor>)
            // {
            //     return gko::CudaExecutor::create(
            //         Kokkos::device_id(), gko::ReferenceExecutor::create(),
            //         std::make_shared<gko::CudaAllocator>(), ex.cuda_stream());
            // }
            else
            {
                throw std::runtime_error("Unsupported executor type");
            }
        },
        exec
    );
}

}
