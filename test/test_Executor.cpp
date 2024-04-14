// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

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
    NeoFOAM::executor exec_0 = GENERATE(
        NeoFOAM::executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::executor(NeoFOAM::OMPExecutor {}),
        NeoFOAM::executor(NeoFOAM::GPUExecutor {})
    );

    NeoFOAM::executor exec_1 = GENERATE(
        NeoFOAM::executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::executor(NeoFOAM::OMPExecutor {}),
        NeoFOAM::executor(NeoFOAM::GPUExecutor {})
    );

    REQUIRE((exec_0.index() == exec_1.index() ? exec_0 == exec_1 : exec_0 != exec_1));
}
