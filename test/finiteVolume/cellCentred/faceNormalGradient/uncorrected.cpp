// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/catch_approx.hpp>

#include "NeoFOAM/NeoFOAM.hpp"


namespace dsl = NeoFOAM::dsl;
namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM
{

template<typename T>
using I = std::initializer_list<T>;

TEMPLATE_TEST_CASE("uncorrected", "[template]", NeoFOAM::scalar, NeoFOAM::Vector)
{
    const size_t nCells = 10;
    NeoFOAM::Executor exec = GENERATE(NeoFOAM::Executor(NeoFOAM::SerialExecutor {})
                                      // NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
                                      // NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    auto mesh = create1DUniformMesh(exec, nCells);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<TestType>>(mesh);

    fvcc::SurfaceField<TestType> phif(exec, "phif", mesh, surfaceBCs);
    fill(phif.internalField(), zero<TestType>::value);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volumeBCs);
    NeoFOAM::parallelFor(
        phi.internalField(),
        KOKKOS_LAMBDA(const size_t i) { return scalar(i + 1) * one<TestType>::value; }
    );
    phi.boundaryField().value() =
        NeoFOAM::Field<TestType>(exec, {0.5 * one<TestType>::value, 10.5 * one<TestType>::value});

    SECTION("Construct from Token" + execName)
    {
        NeoFOAM::Input input = NeoFOAM::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradient<TestType> uncorrected(exec, mesh, input);
    }

    SECTION("faceNormalGrad" + execName)
    {
        NeoFOAM::Input input = NeoFOAM::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradient<TestType> uncorrected(exec, mesh, input);
        uncorrected.faceNormalGrad(phi, phif);

        auto phifHost = phif.internalField().copyToHost();
        auto sPhif = phifHost.span();
        for (size_t i = 0; i < nCells - 1; i++)
        {
            // correct value is 10.0
            REQUIRE(
                NeoFOAM::mag(sPhif[i] - 10.0 * one<TestType>::value)
                == Catch::Approx(0.0).margin(1e-8)
            );
        }
        // left boundary is  -10.0
        REQUIRE(
            NeoFOAM::mag(sPhif[nCells - 1] + 10.0 * one<TestType>::value)
            == Catch::Approx(0.0).margin(1e-8)
        );
        // right boundary is 10.0
        REQUIRE(
            NeoFOAM::mag(sPhif[nCells] - 10.0 * one<TestType>::value)
            == Catch::Approx(0.0).margin(1e-8)
        );
    }
}
}
