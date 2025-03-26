// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

using Operator = NeoFOAM::dsl::Operator;

namespace NeoFOAM
{

TEMPLATE_TEST_CASE("DivOperator", "[template]", NeoFOAM::scalar)
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    auto mesh = create1DUniformMesh(exec, 10);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoFOAM::scalar>>(mesh);

    // compute corresponding uniform faceFlux
    // TODO this should be handled outside of the unit test
    fvcc::SurfaceField<NeoFOAM::scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    NeoFOAM::fill(faceFlux.internalField(), 1.0);
    auto boundFaceFlux = faceFlux.internalField().span();
    // face on the left side has different orientation
    parallelFor(
        exec,
        {mesh.nCells() - 1, mesh.nCells()},
        KOKKOS_LAMBDA(const size_t i) { boundFaceFlux[i] = -1.0; }
    );

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "sf", mesh, volumeBCs);
    NeoFOAM::fill(phi.internalField(), NeoFOAM::one<TestType>::value);
    NeoFOAM::fill(phi.boundaryField().value(), NeoFOAM::one<TestType>::value);
    phi.correctBoundaryConditions();

    auto result = NeoFOAM::Field<TestType>(exec, phi.size());
    NeoFOAM::fill(result, NeoFOAM::zero<TestType>::value);

    SECTION("Construct from Token" + execName)
    {
        NeoFOAM::Input input = NeoFOAM::TokenList({std::string("Gauss"), std::string("linear")});
        fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
    }

    SECTION("Construct from Dictionary" + execName)
    {
        NeoFOAM::Input input = NeoFOAM::Dictionary(
            {{std::string("GradOperator"), std::string("Gauss")},
             {std::string("surfaceInterpolation"), std::string("linear")}}
        );
        auto op = fvcc::GradOperator(Operator::Type::Explicit, faceFlux, phi, input);
        op.grad(result);

        // divergence of a uniform field should be zero
        auto outHost = result.copyToHost();
        for (int i = 0; i < result.size(); i++)
        {
            std::cout << "outHost[" << i << "]" << outHost[i] << "\n";
            REQUIRE(outHost[i] == zero<TestType>::value);
        }
    }
}

}
