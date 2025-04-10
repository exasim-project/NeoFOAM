// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoFOAM/NeoFOAM.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

TEST_CASE("Field<scalar>::addition", "[bench]")
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    DYNAMIC_SECTION("" << size)
    {
        NeoFOAM::Field<NeoFOAM::scalar> cpuA(exec, size);
        NeoFOAM::fill(cpuA, 1.0);
        NeoFOAM::Field<NeoFOAM::scalar> cpuB(exec, size);
        NeoFOAM::fill(cpuB, 2.0);
        NeoFOAM::Field<NeoFOAM::scalar> cpuC(exec, size);
        NeoFOAM::fill(cpuC, 0.0);

        BENCHMARK(std::string(execName)) { return (cpuC = cpuA + cpuB); };
    }
}

TEST_CASE("Field<scalar>::multiplication", "[bench]")
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    DYNAMIC_SECTION("" << size)
    {
        NeoFOAM::Field<NeoFOAM::scalar> cpuA(exec, size);
        NeoFOAM::fill(cpuA, 1.0);
        NeoFOAM::Field<NeoFOAM::scalar> cpuB(exec, size);
        NeoFOAM::fill(cpuB, 2.0);
        NeoFOAM::Field<NeoFOAM::scalar> cpuC(exec, size);
        NeoFOAM::fill(cpuC, 0.0);

        BENCHMARK(std::string(execName)) { return (cpuC = cpuA * cpuB); };
    }
}
