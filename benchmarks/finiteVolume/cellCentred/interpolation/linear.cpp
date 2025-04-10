// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoFOAM/NeoFOAM.hpp"
#include "../../../catch_main.hpp"

#include <catch2/catch_template_test_macros.hpp>

using NeoFOAM::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoFOAM::finiteVolume::cellCentred::VolumeField;
using NeoFOAM::finiteVolume::cellCentred::SurfaceField;

namespace NeoFOAM
{

TEMPLATE_TEST_CASE("linear", "[template]", NeoFOAM::scalar, NeoFOAM::Vector)
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    UnstructuredMesh mesh = create1DUniformMesh(exec, size);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<TestType>>(mesh);
    Input input = TokenList({std::string("linear")});
    auto linear = SurfaceInterpolation<TestType>(exec, mesh, input);

    auto in = VolumeField<TestType>(exec, "in", mesh, {});
    auto out = SurfaceField<TestType>(exec, "out", mesh, surfaceBCs);

    fill(in.internalField(), one<TestType>());

    // capture the value of size as section name
    DYNAMIC_SECTION("" << size)
    {
        BENCHMARK(std::string(execName)) { return (linear.interpolate(in, out)); };
    }
}

}
