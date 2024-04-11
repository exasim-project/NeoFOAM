// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "NeoFOAM/core/executor/executor.hpp"

int main(int argc, char* argv[])
{

    // Initialize Catch2
    Kokkos::initialize(argc, argv);
    Catch::Session session;

    // Specify command line options
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    int result = session.run();

    // Run benchmarks if there are any
    Kokkos::finalize();

    return result;
}

TEST_CASE("Executor Equality")
{
    NeoFOAM::GPUExecutor gpuExec {};
    NeoFOAM::CPUExecutor cpuExec {};
    NeoFOAM::OMPExecutor ompExec {};
    NeoFOAM::executor exec_gpu = gpuExec;
    NeoFOAM::executor exec_cpu = cpuExec;
    NeoFOAM::executor exec_omp = ompExec;

    REQUIRE(exec_gpu == gpuExec);
    REQUIRE(exec_gpu != cpuExec);
    REQUIRE(exec_gpu != ompExec);

    REQUIRE(exec_cpu != gpuExec);
    REQUIRE(exec_cpu == cpuExec);
    REQUIRE(exec_cpu != ompExec);

    REQUIRE(exec_omp != gpuExec);
    REQUIRE(exec_omp != cpuExec);
    REQUIRE(exec_omp == ompExec);
}