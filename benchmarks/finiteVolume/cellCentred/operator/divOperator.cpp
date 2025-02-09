// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoFOAM/NeoFOAM.hpp"
#include "../../../catch_main.hpp"

using Operator = NeoFOAM::dsl::Operator;

TEST_CASE("DivOperator::div", "[bench]")
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::create1DUniformMesh(exec, size);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoFOAM::scalar>>(mesh);
    fvcc::SurfaceField<NeoFOAM::scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    NeoFOAM::fill(faceFlux.internalField(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoFOAM::scalar>>(mesh);
    fvcc::VolumeField<NeoFOAM::scalar> phi(exec, "vf", mesh, volumeBCs);
    fvcc::VolumeField<NeoFOAM::scalar> divPhi(exec, "divPhi", mesh, volumeBCs);
    NeoFOAM::fill(phi.internalField(), 1.0);

    // capture the value of size as section name
    DYNAMIC_SECTION("" << size)
    {
        NeoFOAM::Input input = NeoFOAM::TokenList({std::string("Gauss"), std::string("linear")});
        auto op = fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);

        BENCHMARK(std::string(execName)) { return (op.div(divPhi)); };
    }
}
