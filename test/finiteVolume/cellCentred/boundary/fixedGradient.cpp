// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/boundary/fixedGradient.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/core/dictionary.hpp"

TEST_CASE("fixedValue")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::OMPExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("TestDerivedClass" + execName)
    {
        auto mesh = NeoFOAM::createSingleCellMesh();
        NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, mesh);
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
    }
}
