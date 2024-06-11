// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

TEST_CASE("boundaryField")
{
    using namespace NeoFOAM;
    using BoundaryField = finiteVolume::cellCentred::BoundaryField<scalar>;

    Executor exec =
        GENERATE(Executor(CPUExecutor {}), Executor(OMPExecutor {}), Executor(GPUExecutor {}));

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("can instantiate empty BoundaryField on: " + execName)
    {
        auto bf = BoundaryField(exec);
        REQUIRE(sf.exec() == exec);
    }
}
