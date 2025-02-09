// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

using NeoFOAM::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoFOAM::finiteVolume::cellCentred::VolumeField;
using NeoFOAM::finiteVolume::cellCentred::SurfaceField;

TEMPLATE_TEST_CASE("linear", "", NeoFOAM::scalar, NeoFOAM::Vector)
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    auto mesh = NeoFOAM::create1DUniformMesh(exec, 10);
    NeoFOAM::Input input = NeoFOAM::TokenList({std::string("linear")});
    auto linear = SurfaceInterpolation(exec, mesh, input);

    auto in = VolumeField<TestType>(exec, "in", mesh, {});
    auto out = SurfaceField<TestType>(exec, "out", mesh, {});

    fill(in.internalField(), NeoFOAM::one<TestType>::value);

    linear.interpolate(in, out);

    for (int i = 0; i < out.internalField().size(); i++)
    {
        REQUIRE(out.internalField()[i] == NeoFOAM::one<TestType>::value);
    }
}
