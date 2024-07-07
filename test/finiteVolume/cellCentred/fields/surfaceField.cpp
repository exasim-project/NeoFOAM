// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/fields/geometricField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/surfaceField.hpp"

TEST_CASE("surfaceField")
{
    using namespace NeoFOAM;
    using SurfaceField = finiteVolume::cellCentred::SurfaceField<scalar>;

    Executor exec =
        GENERATE(Executor(CPUExecutor {}), Executor(OMPExecutor {}), Executor(GPUExecutor {}));

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("can instantiate empty surfaceField on: " + execName)
    {
        auto sf = SurfaceField(exec);
        REQUIRE(sf.exec() == exec);
    }
}
