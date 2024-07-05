// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
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
        NeoFOAM::scalar uniformValue {10};
        NeoFOAM::Dictionary dict;
        dict.insert("fixedGradient", uniformValue);
        auto fixedValueBoundary =
            NeoFOAM::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoFOAM::scalar>::create(
                "fixedGradient", mesh, dict, 0
            );

        fixedValueBoundary->correctBoundaryCondition(domainField);

        auto refValues = domainField.boundaryField().refValue().copyToHost();

        for (auto value : refValues.span(std::pair<size_t, size_t> {
                 fixedValueBoundary->patchStart(), fixedValueBoundary->patchEnd()
             }))
        {
            REQUIRE(value == uniformValue);
        }
    }
}
