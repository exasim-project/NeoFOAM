// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoFOAM/NeoFOAM.hpp"


using NeoFOAM::finiteVolume::cellCentred::SurfaceInterpolation;
namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

TEST_CASE("SurfaceInterpolation")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    std::string interpolation = GENERATE(std::string("linear"), std::string("upwind"));

    SECTION("Construct from Token" + execName)
    {
        auto mesh = NeoFOAM::createSingleCellMesh(exec);
        NeoFOAM::Input input = NeoFOAM::TokenList({interpolation});
        fvcc::SurfaceInterpolation<NeoFOAM::scalar> surfInterpolation(exec, mesh, input);
    }

    SECTION("Construct from Dictionary" + execName)
    {
        auto mesh = NeoFOAM::createSingleCellMesh(exec);
        NeoFOAM::Input input =
            NeoFOAM::Dictionary({{std::string("surfaceInterpolation"), interpolation}});
        fvcc::SurfaceInterpolation<NeoFOAM::scalar> surfInterpolation(exec, mesh, input);
    }
}
