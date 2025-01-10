// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

TEST_CASE("fixedValue")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("TestDerivedClass" + execName)
    {
        auto mesh = NeoFOAM::createSingleCellMesh(exec);
        NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, mesh);
        NeoFOAM::scalar setValue {10};
        NeoFOAM::Dictionary dict;
        dict.insert("fixedValue", setValue);
        auto boundary =
            NeoFOAM::finiteVolume::cellCentred::SurfaceBoundaryFactory<NeoFOAM::scalar>::create(
                "fixedValue", mesh, dict, 0
            );

        boundary->correctBoundaryCondition(domainField);

        auto refValues = domainField.boundaryField().refValue().copyToHost();

        for (auto& boundaryValue : refValues.span(boundary->range()))
        {
            REQUIRE(boundaryValue == setValue);
        }

        auto otherBoundary =
            NeoFOAM::finiteVolume::cellCentred::SurfaceBoundaryFactory<NeoFOAM::scalar>::create(
                "fixedValue", mesh, dict, 1
            );

        for (auto& boundaryValue : refValues.span(otherBoundary->range()))
        {
            REQUIRE(boundaryValue != setValue);
        }
    }
}
