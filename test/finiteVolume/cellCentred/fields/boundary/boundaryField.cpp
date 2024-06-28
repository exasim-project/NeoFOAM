// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

using ScalarVolumeBoundaryFactory = fvcc::VolumeBoundaryFactory<NeoFOAM::scalar>;
using VectorVolumeBoundaryFactory = fvcc::VolumeBoundaryFactory<NeoFOAM::Vector>;

class TestDerivedClass : public ScalarVolumeBoundaryFactory
{

public:

    TestDerivedClass(
        const NeoFOAM::UnstructuredMesh& mesh, const NeoFOAM::Dictionary& dict, size_t patchID
    )
        : ScalarVolumeBoundaryFactory(mesh, patchID),
          testString_(dict.get<std::string>("testName")), testValue_(dict.get<double>("testValue"))
    {
        registerClass<TestDerivedClass>();
    }

    static std::unique_ptr<ScalarVolumeBoundaryFactory> create(
        const NeoFOAM::UnstructuredMesh& mesh, const NeoFOAM::Dictionary& dict, std::size_t patchID
    )
    {
        return std::make_unique<TestDerivedClass>(mesh, dict, patchID);
    }

    virtual void correctBoundaryCondition(NeoFOAM::DomainField<NeoFOAM::scalar>& domainField
    ) override
    {
        std::cout << "Correcting boundary conditions" << std::endl;
    }

    static std::string name() { return "TestDerivedClass"; }

private:

    std::string testString_;
    double testValue_;
};

TEST_CASE("boundaryField")
{
    std::cout << "Number of registered classes: " << ScalarVolumeBoundaryFactory::size()
              << std::endl;
    REQUIRE(ScalarVolumeBoundaryFactory::classMap().size() == 5);
    REQUIRE(VectorVolumeBoundaryFactory::classMap().size() == 4);

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::OMPExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("TestDerivedClass" + execName)
    {
        NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, 10, 10, 1);
        NeoFOAM::Dictionary dict;
        dict.insert("testName", "TestDerivedClass");
        dict.insert("testValue", 10);

        auto mesh = NeoFOAM::createSingleCellMesh();

        TestDerivedClass testDerived(mesh, dict, 1);
        testDerived.correctBoundaryCondition(domainField);
        REQUIRE(ScalarVolumeBoundaryFactory::size() == 5);
    }

    SECTION("Can set fixedValue on " + execName)
    {
        auto mesh = NeoFOAM::createSingleCellMesh();
        NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, mesh);
        NeoFOAM::Dictionary dict;
        dict.insert("uniformValue", 1.0);

        fvcc::FixedValue<NeoFOAM::scalar> fixedValueI(mesh, dict, 1);
        fixedValueI.correctBoundaryCondition(domainField);

        auto refValueHost =
            domainField.boundaryField().refValue().copyToHost().field(fixedValueI.patchRange());

        REQUIRE(fixedValueI.patchRange().first == 0);
        REQUIRE(fixedValueI.patchRange().second == 1);

        for (auto value : refValueHost)
        {
            REQUIRE(value == 1.0);
        }
    }
}
