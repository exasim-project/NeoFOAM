// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "NeoN/NeoN.hpp"

#include <string>
#include <utility>

// This class shows how to implement a simple generator for Catch tests
class ExecutorGenerator final :
    public Catch::Generators::IGenerator<std::pair<std::string, NeoN::Executor>>
{
public:

    std::size_t i = 0;
    std::vector<std::pair<std::string, NeoN::Executor>> execs {};
    std::pair<std::string, NeoN::Executor> current_exec = {
        "SerialExecutor",
        NeoN::SerialExecutor {}
    };

    ExecutorGenerator()
    {
#if defined(NEON_ENABLE_OPENMP)
        execs.push_back({"CPUExecutor", NeoN::CPUExecutor {}});
#elif defined(NEON_ENABLE_THREADS)
        execs.push_back({"CPUExecutor", NeoN::CPUExecutor {}});
#endif
#if defined(NEON_ENABLE_CUDA)
        execs.push_back({"GPUExecutor", NeoN::GPUExecutor {}});
#elif defined(NEON_ENABLE_HIP)
        execs.push_back({"GPUExecutor", NeoN::GPUExecutor {}});
#elif defined(NEON_ENABLE_SYCL)
        execs.push_back({"GPUExecutor", NeoN::GPUExecutor {}});
#endif
    }

    std::pair<std::string, NeoN::Executor> const& get() const override;
    bool next() override
    {
        if (i >= execs.size()) return false;
        current_exec = execs[i];
        i++;
        return true;
    }
};

// Avoids -Wweak-vtables
std::pair<std::string, NeoN::Executor> const& ExecutorGenerator::get() const
{
    return current_exec;
}

Catch::Generators::GeneratorWrapper<std::pair<std::string, NeoN::Executor>> allAvailableExecutor()
{
    return Catch::Generators::GeneratorWrapper<std::pair<std::string, NeoN::Executor>>(
        Catch::Detail::make_unique<ExecutorGenerator>()
    );
}

// Ginkgo's SYCL backend does not implement the distributed index map (build_mapping and
// map_to_local are GKO_NOT_IMPLEMENTED), so the distributed solves cannot run on the SYCL GPU
// executor. Leave it out rather than SKIP: a SKIP marks the whole test case as skipped, and Catch2
// then fails the run because no test case passed.
Catch::Generators::GeneratorWrapper<std::pair<std::string, NeoN::Executor>>
distributedSolverExecutor()
{
#if defined(NEON_ENABLE_SYCL)
    return Catch::Generators::filter(
        [](const std::pair<std::string, NeoN::Executor>& exec)
        { return !std::holds_alternative<NeoN::GPUExecutor>(exec.second); },
        allAvailableExecutor()
    );
#else
    return allAvailableExecutor();
#endif
}
