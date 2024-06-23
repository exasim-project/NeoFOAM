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

    TestDerivedClass(std::string name, double test)
        : ScalarVolumeBoundaryFactory(), testString_(name), testValue_(test)
    {
        registerClass<TestDerivedClass>();
    }

    static std::unique_ptr<ScalarVolumeBoundaryFactory>
    create(const NeoFOAM::UnstructuredMesh& mesh, const NeoFOAM::Dictionary& dict, int patchID)
    {
        std::string name = dict.get<std::string>("name");
        double test = dict.get<double>("test");
        return std::make_unique<TestDerivedClass>(name, test);
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

        TestDerivedClass testDerived("TestDerivedClass", 1.0);
        testDerived.correctBoundaryCondition(domainField);
        REQUIRE(ScalarVolumeBoundaryFactory::size() == 5);
    }

    SECTION("FixedValue" + execName)
    {
        NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, 10, 10, 1);

        fvcc::FixedValue<NeoFOAM::scalar> fixedValue(0, 10, 1, 1.0);
        fixedValue.correctBoundaryCondition(domainField);
        auto refValueHost = domainField.boundaryField().refValue().copyToHost().field();
        for (std::size_t i = 0; i < 10; i++)
        {
            REQUIRE(refValueHost[i] == 1.0);
        }
    }
}
