// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/boundary.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

using Operator = NeoFOAM::dsl::Operator;

TEST_CASE("DivOperator")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);
    // FIXME: take 1d mesh
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoFOAM::scalar>>(mesh);

    fvcc::SurfaceField<NeoFOAM::scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    NeoFOAM::fill(faceFlux.internalField(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoFOAM::scalar>>(mesh);
    fvcc::VolumeField<NeoFOAM::scalar> phi(exec, "sf", mesh, volumeBCs);
    NeoFOAM::fill(phi.internalField(), 1.0);

    SECTION("Construct from Token" + execName)
    {
        auto mesh = NeoFOAM::createSingleCellMesh(exec);
        NeoFOAM::Input input = NeoFOAM::TokenList({std::string("Gauss"), std::string("linear")});
        fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
    }

    SECTION("Construct from Dictionary" + execName)
    {
        auto mesh = NeoFOAM::createSingleCellMesh(exec);
        NeoFOAM::Input input = NeoFOAM::Dictionary(
            {{std::string("DivOperator"), std::string("Gauss")},
             {std::string("surfaceInterpolation"), std::string("linear")}}
        );
        fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
    }
}
