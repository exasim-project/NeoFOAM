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
    NeoFOAM::Executor cpuExec0(NeoFOAM::SerialExecutor {});
    NeoFOAM::Executor ompExec0(NeoFOAM::OMPExecutor {});
    NeoFOAM::Executor gpuExec0(NeoFOAM::GPUExecutor {});

    NeoFOAM::Executor cpuExec1(NeoFOAM::SerialExecutor {});
    NeoFOAM::Executor ompExec1(NeoFOAM::OMPExecutor {});
    NeoFOAM::Executor gpuExec1(NeoFOAM::GPUExecutor {});

    REQUIRE(cpuExec0 == cpuExec1);
    REQUIRE(cpuExec0 != ompExec1);
    REQUIRE(cpuExec0 != gpuExec1);

    REQUIRE(ompExec0 != cpuExec1);
    REQUIRE(ompExec0 == ompExec1);
    REQUIRE(ompExec0 != gpuExec1);

    REQUIRE(gpuExec0 != cpuExec1);
    REQUIRE(gpuExec0 != ompExec1);
    REQUIRE(gpuExec0 == gpuExec1);
}
