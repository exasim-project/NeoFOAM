// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoFOAM/core/executor/executor.hpp"

TEST_CASE("Executor Equality")
{
    NeoFOAM::executor cpu_exec_0(NeoFOAM::CPUExecutor {});
    NeoFOAM::executor omp_exec_0(NeoFOAM::OMPExecutor {});
    NeoFOAM::executor gpu_exec_0(NeoFOAM::GPUExecutor {});

    NeoFOAM::executor cpu_exec_1(NeoFOAM::CPUExecutor {});
    NeoFOAM::executor omp_exec_1(NeoFOAM::OMPExecutor {});
    NeoFOAM::executor gpu_exec_1(NeoFOAM::GPUExecutor {});

    REQUIRE(cpu_exec_0 == cpu_exec_1);
    REQUIRE(cpu_exec_0 != omp_exec_1);
    REQUIRE(cpu_exec_0 != gpu_exec_1);

    REQUIRE(omp_exec_0 != cpu_exec_1);
    REQUIRE(omp_exec_0 == omp_exec_1);
    REQUIRE(omp_exec_0 != gpu_exec_1);

    REQUIRE(gpu_exec_0 != cpu_exec_1);
    REQUIRE(gpu_exec_0 != omp_exec_1);
    REQUIRE(gpu_exec_0 == gpu_exec_1);
}
