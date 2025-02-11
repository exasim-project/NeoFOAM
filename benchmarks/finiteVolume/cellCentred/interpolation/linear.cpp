// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoFOAM/NeoFOAM.hpp"
#include "../../../catch_main.hpp"

using NeoFOAM::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoFOAM::finiteVolume::cellCentred::VolumeField;
using NeoFOAM::finiteVolume::cellCentred::SurfaceField;
using NeoFOAM::Input;

TEST_CASE("linear", "[bench]")
{
    using TestType = NeoFOAM::scalar;
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::create1DUniformMesh(exec, size);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoFOAM::scalar>>(mesh);
    Input input = NeoFOAM::TokenList({std::string("linear")});
    auto linear = SurfaceInterpolation(exec, mesh, input);

    auto in = VolumeField<TestType>(exec, "in", mesh, {});
    auto out = SurfaceField<TestType>(exec, "out", mesh, surfaceBCs);

    fill(in.internalField(), NeoFOAM::one<NeoFOAM::scalar>::value);

    // capture the value of size as section name
    DYNAMIC_SECTION("" << size)
    {
        BENCHMARK(std::string(execName)) { return (linear.interpolate(in, out)); };
    }
}
