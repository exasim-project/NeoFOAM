// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "catch2_common.hpp"

#include <Kokkos_Core.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

#include <string>
#include <utility>

// This class shows how to implement a simple generator for Catch tests
class ExecutorGenerator final :
    public Catch::Generators::IGenerator<std::pair<std::string, NeoFOAM::Executor>>
{
public:

    int i = 0;
    std::vector<std::pair<std::string, NeoFOAM::Executor>> execs {};
    std::pair<std::string, NeoFOAM::Executor> current_exec = {
        "SerialExecutor", NeoFOAM::SerialExecutor {}
    };

    ExecutorGenerator()
    {
#if defined(KOKKOS_ENABLE_OPENMP)
        execs.push_back({"CPUExecutor", NeoFOAM::CPUExecutor {}});
#elif defined(KOKKOS_ENABLE_THREADS)
        execs.push_back({"CPUExecutor", NeoFOAM::CPUExecutor {}});
#endif

        execs.push_back({"GPUExecutor", NeoFOAM::GPUExecutor {}});
#if defined(KOKKOS_ENABLE_CUDA)
        execs.push_back({"GPUExecutor", NeoFOAM::GPUExecutor {}});
#elif defined(KOKKOS_ENABLE_HIP)
        execs.push_back({"GPUExecutor", NeoFOAM::GPUExecutor {}});
#elif defined(KOKKOS_ENABLE_SYCL)
        execs.push_back({"GPUExecutor", NeoFOAM::GPUExecutor {}});
#endif
    }

    std::pair<std::string, NeoFOAM::Executor> const& get() const override;
    bool next() override
    {
        if (i >= execs.size()) return false;
        current_exec = execs[i];
        i++;
        return true;
    }
};

// Avoids -Wweak-vtables
std::pair<std::string, NeoFOAM::Executor> const& ExecutorGenerator::get() const
{
    return current_exec;
}

Catch::Generators::GeneratorWrapper<std::pair<std::string, NeoFOAM::Executor>>
allAvailableExecutor()
{
    return Catch::Generators::GeneratorWrapper<std::pair<std::string, NeoFOAM::Executor>>(
        Catch::Detail::make_unique<ExecutorGenerator>()
    );
}
