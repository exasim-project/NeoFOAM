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

using ScalarBoundary = fvcc::BoundaryFactory<NeoFOAM::scalar, int>;
using VectorBoundary = fvcc::BoundaryFactory<NeoFOAM::Vector, int>;

// NOTE the second template type for now is just a dummy
template<typename ValueType>
class FixedValue :
    public fvcc::BoundaryFactory<ValueType, ValueType>,
    public fvcc::BoundaryCorrectionStrategy<ValueType>
{

public:

    using FixedValueType = FixedValue<ValueType>;

    template<typename executor>
    void setFixedValue(const executor& exec, std::span<ValueType> field, ValueType value)
    {
        if constexpr (std::is_same<std::remove_reference_t<executor>, NeoFOAM::CPUExecutor>::value)
        {
            for (std::size_t i = this->start_; i < this->end_; i++)
            {
                field[i] = value;
            }
        }
        else
        {
            using runOn = typename executor::exec;
            Kokkos::parallel_for(
                "parallelForImpl",
                Kokkos::RangePolicy<runOn>(this->start_, this->end_),
                KOKKOS_LAMBDA(std::size_t i) { field[i] = value; }
            );
        }
    }

    FixedValue(
        std::shared_ptr<const NeoFOAM::UnstructuredMesh> mesh,
        std::size_t patchID,
        ValueType uniformValue
    )
        : fvcc::BoundaryFactory<ValueType, ValueType>(mesh, patchID), uniformValue_(uniformValue)
    {
        // fvcc::BoundaryFactory<ValueType, ValueType>::template registerClass<FixedValueType>();
        // setCorrectionStrategy(std::unique_ptr<fvcc::BoundaryCorrectionStrategy<ValueType>>(this));
    }

    static std::unique_ptr<FixedValueType> create(
        std::shared_ptr<const NeoFOAM::UnstructuredMesh> mesh, int patchID, ValueType uniformValue
    )
    {
        return std::make_unique<FixedValueType>(mesh, patchID, uniformValue);
    }

    virtual void correctBoundaryConditionsImpl(NeoFOAM::DomainField<ValueType>& domainField
    ) override
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
};

template class FixedValue<NeoFOAM::scalar>;
// template class FixedValue<NeoFOAM::Vector>;

TEST_CASE("boundaryField")
{
    std::cout << "Number of registered classes: " << ScalarBoundary::nRegistered() << std::endl;
    REQUIRE(ScalarBoundary::classMap().size() == 2);
    REQUIRE(VectorBoundary::classMap().size() == 1);

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::OMPExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    //   SECTION("TestDerivedClass" + execName)
    //   {
    //       NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, 10, 10, 1);
    //
    //       TestDerivedClass testDerived("TestDerivedClass", 1.0);
    //       testDerived.correctBoundaryConditions(domainField);
    //       REQUIRE(ScalarBoundary::nRegistered() == 2);
    //   }

    //   SECTION("fixedValue" + execName)
    //   {
    //       NeoFOAM::DomainField<NeoFOAM::scalar> domainField(exec, 10, 10, 1);
    //
    //       FixedValue<NeoFOAM::scalar> fixedValue(0, 10, 1, 1.0);
    //       FixedValue.correctBoundaryConditions(domainField);
    //       auto refValueHost = domainField.boundaryField().refValue().copyToHost().field();
    //       for (std::size_t i = 0; i < 10; i++)
    //       {
    //           REQUIRE(refValueHost[i] == 1.0);
    //       }
    //   }
}
