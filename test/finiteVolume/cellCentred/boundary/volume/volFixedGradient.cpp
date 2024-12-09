// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/boundary/volume/fixedGradient.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/core/dictionary.hpp"

#include "NeoFOAM/finiteVolume/cellCentred/boundary.hpp"

TEST_CASE("fixedGradient")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("TestDerivedClass" + execName)
    {
        // unit cube mesh
        auto mesh = NeoFOAM::createSingleCellMesh(exec);
        // the same as (exec, mesh.nCells(), mesh.nBoundaryFaces(), mesh.nBoundaries())
        NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, mesh);
        NeoFOAM::fill(domainField.internalField(), 1.0);
        NeoFOAM::fill(domainField.boundaryField().refGrad(), -1.0);
        NeoFOAM::fill(domainField.boundaryField().refValue(), -1.0);
        NeoFOAM::fill(domainField.boundaryField().valueFraction(), -1.0);
        NeoFOAM::fill(domainField.boundaryField().value(), -1.0);

        SECTION("zeroGradient")
        {
            NeoFOAM::scalar setValue {0};
            NeoFOAM::Dictionary dict;
            dict.insert("fixedGradient", setValue);
            auto boundary =
                NeoFOAM::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoFOAM::scalar>::create(
                    "fixedGradient", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(domainField);

            auto refValues = domainField.boundaryField().refGrad().copyToHost();

            for (auto boundaryValue : refValues.span(boundary->range()))
            {
                REQUIRE(boundaryValue == setValue);
            }

            auto values = domainField.boundaryField().value().copyToHost();

            for (auto& boundaryValue : values.span(boundary->range()))
            {
                REQUIRE(boundaryValue == 1.0);
            }
        }

        SECTION("FixedGradient_10")
        {
            NeoFOAM::scalar setValue {10};
            NeoFOAM::Dictionary dict;
            dict.insert("fixedGradient", setValue);
            auto boundary =
                NeoFOAM::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoFOAM::scalar>::create(
                    "fixedGradient", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(domainField);

            auto refValues = domainField.boundaryField().refGrad().copyToHost();

            for (auto boundaryValue : refValues.span(boundary->range()))
            {
                REQUIRE(boundaryValue == setValue);
            }

            auto values = domainField.boundaryField().value().copyToHost();

            // deltaCoeffs is the inverse distance and has a value of 2.0
            // so the value is 1.0 + 10 / 2.0 = 6.0
            for (auto& boundaryValue : values.span(boundary->range()))
            {
                REQUIRE(boundaryValue == 6.0);
            }
        }
    }
}
