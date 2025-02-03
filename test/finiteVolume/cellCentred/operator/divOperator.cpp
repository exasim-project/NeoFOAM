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


TEMPLATE_TEST_CASE("DivOperator", "[template]", NeoFOAM::scalar, NeoFOAM::Vector)
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    // TODO take 1d mesh
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoFOAM::scalar>>(mesh);

    fvcc::SurfaceField<NeoFOAM::scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    NeoFOAM::fill(faceFlux.internalField(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> vecPhi(exec, "sf", mesh, volumeBCs);
    // NeoFOAM::fill(vecPhi.internalField(), NeoFOAM::Vector{1.0, 1.0, 1.0});

    // SECTION("Construct from Token" + execName)
    // {
    //     NeoFOAM::Input input = NeoFOAM::TokenList({std::string("Gauss"), std::string("linear")});
    //     fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
    // }

    // SECTION("Construct from Dictionary" + execName)
    // {
    //     NeoFOAM::Input input = NeoFOAM::Dictionary(
    //         {{std::string("DivOperator"), std::string("Gauss")},
    //          {std::string("surfaceInterpolation"), std::string("linear")}}
    //     );
    //     fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
    // }
}
