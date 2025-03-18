// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include <Kokkos_Core.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

// This class shows how to implement a simple generator for Catch tests
class ExecutorGenerator final : public Catch::Generators::IGenerator<NeoFOAM::Executor>
{
public:

    int i = 0;
    std::vector<NeoFOAM::Executor> execs {};
    NeoFOAM::Executor current_exec = NeoFOAM::SerialExecutor {};

    ExecutorGenerator()
    {
#if defined(KOKKOS_ENABLE_OPENMP)
        execs.push_back(NeoFOAM::CPUExecutor {});
#elif defined(KOKKOS_ENABLE_THREADS)
        execs.push_back(NeoFOAM::CPUExecutor {});
#endif

#if defined(KOKKOS_ENABLE_CUDA)
        execs.push_back(NeoFOAM::GPUExecutor {});
#elif defined(KOKKOS_ENABLE_HIP)
        execs.push_back(NeoFOAM::GPUExecutor {});
#elif defined(KOKKOS_ENABLE_SYCL)
        execs.push_back(NeoFOAM::GPUExecutor {});
#endif
    }

    NeoFOAM::Executor const& get() const override;
    bool next() override
    {
        if (i >= execs.size()) return false;
        current_exec = execs[i];
        i++;
        return true;
    }
};

// Avoids -Wweak-vtables
NeoFOAM::Executor const& ExecutorGenerator::get() const { return current_exec; }

Catch::Generators::GeneratorWrapper<NeoFOAM::Executor> allAvailableExecutor()
{
    return Catch::Generators::GeneratorWrapper<NeoFOAM::Executor>(
        Catch::Detail::make_unique<ExecutorGenerator>()
    );
}
