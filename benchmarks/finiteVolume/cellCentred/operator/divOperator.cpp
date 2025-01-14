// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <vector>
#include <iostream>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

using Operator = NeoFOAM::dsl::Operator;

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

TEST_CASE("divOperator", "[bench]")
{

    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    CAPTURE(size); // Capture the value of size

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
        {
            NeoFOAM::Input input =
                NeoFOAM::TokenList({std::string("Gauss"), std::string("linear")});
            auto op = fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);

            BENCHMARK("DivOperator<scalar>::addition on " + execName) { return (op.div(divPhi)); };
        }
    }
}
