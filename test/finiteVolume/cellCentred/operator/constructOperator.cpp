// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core/inputs.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

TEST_CASE("DivOperator")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("Construct from Token" + execName)
    {
        auto mesh = NeoFOAM::createSingleCellMesh(exec);
        NeoFOAM::Input input = NeoFOAM::TokenList({std::string("Gauss"), std::string("linear")});
        fvcc::DivOperator(exec, mesh, input);
    }

    SECTION("Construct from Dictionary" + execName)
    {
        auto mesh = NeoFOAM::createSingleCellMesh(exec);
        NeoFOAM::Input input = NeoFOAM::Dictionary(
            {{std::string("DivOperator"), std::string("Gauss")},
             {std::string("surfaceInterpolation"), std::string("linear")}}
        );
        fvcc::DivOperator(exec, mesh, input);
    }
}

TEST_CASE("SurfaceInterpolation")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    std::string interpolation = GENERATE(std::string("linear"), std::string("upwind"));

    SECTION("Construct from Token" + execName)
    {
        auto mesh = NeoFOAM::createSingleCellMesh(exec);
        NeoFOAM::Input input = NeoFOAM::TokenList({interpolation});
        fvcc::SurfaceInterpolation surfInterpolation(exec, mesh, input);
    }

    SECTION("Construct from Dictionary" + execName)
    {
        auto mesh = NeoFOAM::createSingleCellMesh(exec);
        NeoFOAM::Input input =
            NeoFOAM::Dictionary({{std::string("surfaceInterpolation"), interpolation}});
        fvcc::SurfaceInterpolation surfInterpolation(exec, mesh, input);
    }
}
