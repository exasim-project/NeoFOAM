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

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

using ScalarVolumeBoundaryFactory = fvcc::VolumeBoundaryFactory<NeoFOAM::scalar>;
using VectorVolumeBoundaryFactory = fvcc::VolumeBoundaryFactory<NeoFOAM::Vector>;


TEST_CASE("boundaryField")
{
    std::cout << "Number of registered classes: " << ScalarVolumeBoundaryFactory::size()
              << std::endl;
    REQUIRE(ScalarVolumeBoundaryFactory::classMap().size() == 2);
    REQUIRE(VectorVolumeBoundaryFactory::classMap().size() == 1);

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::OMPExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("TestDerivedClass" + execName)
    {
        NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, 10, 10, 1);

        TestDerivedClass testDerived("TestDerivedClass", 1.0);
        testDerived.correctBoundaryCondition(domainField);
        REQUIRE(ScalarVolumeBoundaryFactory::size() == 2);
    }

    SECTION("FixedValue" + execName)
    {
        NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, 10, 10, 1);

        FixedValue<NeoFOAM::scalar> fixedValue(0, 10, 1, 1.0);
        fixedValue.correctBoundaryCondition(domainField);
        auto refValueHost = domainField.boundaryField().refValue().copyToHost().field();
        for (std::size_t i = 0; i < 10; i++)
        {
            REQUIRE(refValueHost[i] == 1.0);
        }
    }
}
