// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/catch_approx.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

using NeoFOAM::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoFOAM::finiteVolume::cellCentred::VolumeField;
using NeoFOAM::finiteVolume::cellCentred::SurfaceField;

namespace NeoFOAM
{

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("uncorrected")
{
    const size_t nCells = 10;
    NeoFOAM::Executor exec = GENERATE(NeoFOAM::Executor(NeoFOAM::SerialExecutor {})
                                      // NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
                                      // NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    auto mesh = create1DUniformMesh(exec, nCells);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoFOAM::scalar>>(mesh);

    fvcc::SurfaceField<NeoFOAM::scalar> phif(exec, "phif", mesh, surfaceBCs);
    fill(phif.internalField(), 0.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoFOAM::scalar>>(mesh);
    fvcc::VolumeField<NeoFOAM::scalar> phi(exec, "phi", mesh, volumeBCs);
    NeoFOAM::parallelFor(
        phi.internalField(), KOKKOS_LAMBDA(const size_t i) { return scalar(i + 1); }
    );
    phi.boundaryField().value() = NeoFOAM::Field<NeoFOAM::scalar>(exec, {0.5, 10.5});

    SECTION("Construct from Token" + execName)
    {
        NeoFOAM::Input input = NeoFOAM::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradient<NeoFOAM::scalar> uncorrected(exec, mesh, input);
    }

    SECTION("faceNormalGrad" + execName)
    {
        NeoFOAM::Input input = NeoFOAM::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradient<NeoFOAM::scalar> uncorrected(exec, mesh, input);
        uncorrected.faceNormalGrad(phi, phif);

        auto phifHost = phif.internalField().copyToHost();
        auto sPhif = phifHost.span();
        for (size_t i = 0; i < nCells - 1; i++)
        {
            REQUIRE(sPhif[i] == Catch::Approx(10.0).margin(1e-8));
        }
        // left boundary
        REQUIRE(sPhif[nCells - 1] == Catch::Approx(-10.0).margin(1e-8));
        // right boundary
        REQUIRE(sPhif[nCells] == Catch::Approx(10.0).margin(1e-8));
    }
}
}
