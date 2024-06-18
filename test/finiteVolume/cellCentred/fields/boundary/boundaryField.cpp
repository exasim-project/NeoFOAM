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

using ScalarVolumeBoundaryModel = fvcc::VolumeBoundaryModel<NeoFOAM::scalar>;
using VectorVolumeBoundaryModel = fvcc::VolumeBoundaryModel<NeoFOAM::Vector>;

class TestDerivedClass : public ScalarVolumeBoundaryModel
{

public:

    TestDerivedClass(std::string name, double test)
        : ScalarVolumeBoundaryModel(), testString_(name), testValue_(test)
    {
        registerClass<TestDerivedClass>();
    }

    static std::unique_ptr<ScalarVolumeBoundaryModel>
    create(const NeoFOAM::UnstructuredMesh& mesh, const NeoFOAM::Dictionary& dict, int patchID)
    {
        std::string name = dict.get<std::string>("name");
        double test = dict.get<double>("test");
        return std::make_unique<TestDerivedClass>(name, test);
    }

    virtual void correctBoundaryConditions(NeoFOAM::DomainField<NeoFOAM::scalar>& domainField
    ) override
    {
        std::cout << "Correcting boundary conditions" << std::endl;
    }

    static std::string name() { return "TestDerivedClass"; }

private:

    std::string testString_;
    double testValue_;
};

template<typename ValueType>
class fixedValue : public fvcc::VolumeBoundaryModel<ValueType>
{

public:

    using FixedValueType = fixedValue<ValueType>;

    template<typename executor>
    void setFixedValue(const executor& exec, std::span<ValueType> field, ValueType value)
    {
        if constexpr (std::is_same<std::remove_reference_t<executor>, NeoFOAM::CPUExecutor>::value)
        {
            for (std::size_t i = start_; i < end_; i++)
            {
                field[i] = value;
            }
        }
        else
        {
            using runOn = typename executor::exec;
            Kokkos::parallel_for(
                "parallelForImpl",
                Kokkos::RangePolicy<runOn>(start_, end_),
                KOKKOS_LAMBDA(std::size_t i) { field[i] = value; }
            );
        }
    }

    fixedValue(std::size_t start, std::size_t end, std::size_t patchID, ValueType uniformValue)
        : fvcc::VolumeBoundaryModel<ValueType>(), start_(start), end_(end), patchID_(patchID),
          uniformValue_(uniformValue)
    {
        fvcc::VolumeBoundaryModel<ValueType>::template registerClass<FixedValueType>();
    }

    static std::unique_ptr<fvcc::VolumeBoundaryModel<ValueType>>
    create(const NeoFOAM::UnstructuredMesh& mesh, const NeoFOAM::Dictionary& dict, int patchID)
    {

        ValueType uniformValue = dict.get<ValueType>("uniformValue");
        std::size_t start = dict.get<std::size_t>("start");
        std::size_t end = dict.get<std::size_t>("end");
        return std::make_unique<FixedValueType>(start, end, patchID, uniformValue);
    }

    virtual void correctBoundaryConditions(NeoFOAM::DomainField<ValueType>& domainField) override
    {
        std::visit(
            [&](auto exec)
            { setFixedValue(exec, domainField.boundaryField().refValue().field(), uniformValue_); },
            domainField.exec()
        );
    }

    static std::string name() { return "fixedValue"; }

private:

    ValueType uniformValue_;
    std::size_t start_;
    std::size_t end_;
    std::size_t patchID_;
};

template class fixedValue<NeoFOAM::scalar>;
template class fixedValue<NeoFOAM::Vector>;

TEST_CASE("boundaryField")
{
    std::cout << "Number of registered classes: " << ScalarVolumeBoundaryModel::nRegistered()
              << std::endl;
    REQUIRE(ScalarVolumeBoundaryModel::classMap().size() == 2);
    REQUIRE(VectorVolumeBoundaryModel::classMap().size() == 1);

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
        testDerived.correctBoundaryConditions(domainField);
        REQUIRE(ScalarVolumeBoundaryModel::nRegistered() == 2);
    }

    SECTION("fixedValue" + execName)
    {
        NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, 10, 10, 1);

        fixedValue<NeoFOAM::scalar> fixedValue(0, 10, 1, 1.0);
        fixedValue.correctBoundaryConditions(domainField);
        auto refValueHost = domainField.boundaryField().refValue().copyToHost().field();
        for (std::size_t i = 0; i < 10; i++)
        {
            REQUIRE(refValueHost[i] == 1.0);
        }
    }
}
