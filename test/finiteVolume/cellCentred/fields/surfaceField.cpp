// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

#include "NeoFOAM/finiteVolume/cellCentred/boundary.hpp"

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("surfaceField")
{
    namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("can instantiate SurfaceField with fixedValues on: " + execName)
    {
        NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);
        std::vector<fvcc::SurfaceBoundary<NeoFOAM::scalar>> bcs {};
        for (auto patchi : I<NeoFOAM::size_t> {0, 1, 2, 3})
        {
            NeoFOAM::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 2.0);
            bcs.push_back(fvcc::SurfaceBoundary<NeoFOAM::scalar>(mesh, dict, patchi));
        }

        fvcc::SurfaceField<NeoFOAM::scalar> sf(exec, "sf", mesh, bcs);
        // the internal field is 4 because the mesh has 4 boundaryFaces
        NeoFOAM::fill(sf.internalField(), 1.0);

        {
            auto internalValues = sf.internalField().copyToHost();
            for (size_t i = 0; i < internalValues.size(); ++i)
            {
                REQUIRE(internalValues[i] == 1.0);
            }
        }

        sf.correctBoundaryConditions();
        REQUIRE(sf.exec() == exec);
        REQUIRE(sf.internalField().size() == 4);

        // the correctBoundaryConditions also sets the internalfield to 2.0
        // for surfaceFields
        auto internalValues = sf.internalField().copyToHost();
        for (size_t i = 0; i < internalValues.size(); ++i)
        {
            REQUIRE(internalValues[i] == 2.0);
        }

        auto values = sf.boundaryField().value().copyToHost();
        for (size_t i = 0; i < values.size(); ++i)
        {
            REQUIRE(values[i] == 2.0);
        }

        auto refValue = sf.boundaryField().refValue().copyToHost();
        for (size_t i = 0; i < refValue.size(); ++i)
        {
            REQUIRE(refValue[i] == 2.0);
        }
    }
}
