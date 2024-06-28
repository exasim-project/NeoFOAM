// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core.hpp"

#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"

TEST_CASE("surfaceField")
{
    REQUIRE(false);
    // using namespace NeoFOAM;
    // using VolumeField = finiteVolume::cellCentred::VolumeField<scalar>;

    // Executor exec =
    //     GENERATE(Executor(CPUExecutor {}), Executor(OMPExecutor {}), Executor(GPUExecutor {}));

    // std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    // SECTION("can instantiate empty VolumeField on: " + execName)
    // {
    //     auto vf = VolumeField(exec);
    //     REQUIRE(vf.exec() == exec);
    // }
}
