// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/DSL/coeff.hpp"
#include "NeoFOAM/DSL/operator.hpp"

using Field = NeoFOAM::Field<NeoFOAM::scalar>;
using Coeff = NeoFOAM::DSL::Coeff;
using Operator = NeoFOAM::DSL::Operator;
using OperatorMixin = NeoFOAM::DSL::OperatorMixin;
using Executor = NeoFOAM::Executor;
using VolumeField = fvcc::VolumeField<NeoFOAM::scalar>;
using BoundaryFields = NeoFOAM::BoundaryFields<NeoFOAM::scalar>;
namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

/* A dummy implementation of a Operator
 * following the Operator interface */
class Dummy : public OperatorMixin
{

public:

    Dummy(const Executor& exec, VolumeField& field) : OperatorMixin(exec), field_(field) {}

    void explicitOperation(Field& source) const
    {
        auto sourceSpan = source.span();
        auto fieldSpan = field_.internalField().span();
        auto coeff = getCoefficient();
        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t i) { sourceSpan[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    std::string getName() const { return "Dummy"; }

    const VolumeField& volumeField() const { return field_; }

    VolumeField& volumeField() { return field_; }

    Operator::Type getType() const { return Operator::Type::Explicit; }

    size_t getSize() const { return field_.internalField().size(); }

private:

    VolumeField& field_;
};

NeoFOAM::scalar getField(const NeoFOAM::Field<NeoFOAM::scalar>& source)
{
    auto sourceField = source.copyToHost();
    return sourceField.span()[0];
}
