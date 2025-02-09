// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

using Field = NeoFOAM::Field<NeoFOAM::scalar>;
using Coeff = NeoFOAM::dsl::Coeff;
using Operator = NeoFOAM::dsl::Operator;
using Executor = NeoFOAM::Executor;
using VolumeField = fvcc::VolumeField<NeoFOAM::scalar>;
using OperatorMixin = NeoFOAM::dsl::OperatorMixin<VolumeField>;
using BoundaryFields = NeoFOAM::BoundaryFields<NeoFOAM::scalar>;

/* A dummy implementation of a SpatialOperator
 * following the SpatialOperator interface */
class Dummy : public OperatorMixin
{

public:

    Dummy(VolumeField& field) : OperatorMixin(field.exec(), field, Operator::Type::Explicit) {}

    Dummy(VolumeField& field, Operator::Type type) : OperatorMixin(field.exec(), field, type) {}

    void explicitOperation(Field& source)
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

    void implicitOperation(la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx>& ls)
    {
        auto values = ls.matrix().values();
        auto rhs = ls.rhs().span();
        auto fieldSpan = field_.internalField().span();
        auto coeff = getCoefficient();

        // update diag
        NeoFOAM::parallelFor(
            exec(),
            {0, values.size()},
            KOKKOS_LAMBDA(const size_t i) { values[i] += coeff[i] * fieldSpan[i]; }
        );

        // update rhs
        NeoFOAM::parallelFor(
            exec(),
            ls.rhs().range(),
            KOKKOS_LAMBDA(const size_t i) { rhs[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> createEmptyLinearSystem() const
    {
        NeoFOAM::Field<NeoFOAM::scalar> values(exec(), 1, 0.0);
        NeoFOAM::Field<NeoFOAM::localIdx> colIdx(exec(), 1, 0.0);
        NeoFOAM::Field<NeoFOAM::localIdx> rowPtrs(exec(), {0, 1});
        NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> csrMatrix(
            values, colIdx, rowPtrs
        );

        NeoFOAM::Field<NeoFOAM::scalar> rhs(exec(), 1, 0.0);
        NeoFOAM::la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> linearSystem(
            csrMatrix, rhs, "diagonal"
        );
        return linearSystem;
    }

    std::string getName() const { return "Dummy"; }
};

/* A dummy implementation of a SpatialOperator
 * following the SpatialOperator interface */
class TemporalDummy : public OperatorMixin
{

public:

    TemporalDummy(VolumeField& field) : OperatorMixin(field.exec(), field, Operator::Type::Explicit)
    {}

    TemporalDummy(VolumeField& field, Operator::Type type)
        : OperatorMixin(field.exec(), field, type)
    {}

    void explicitOperation(Field& source, NeoFOAM::scalar t, NeoFOAM::scalar dt)
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

    void implicitOperation(
        la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx>& ls,
        NeoFOAM::scalar t,
        NeoFOAM::scalar dt
    )
    {
        auto values = ls.matrix().values();
        auto rhs = ls.rhs().span();
        auto fieldSpan = field_.internalField().span();
        auto coeff = getCoefficient();

        // update diag
        NeoFOAM::parallelFor(
            exec(),
            {0, values.size()},
            KOKKOS_LAMBDA(const size_t i) { values[i] += coeff[i] * fieldSpan[i]; }
        );

        // update rhs
        NeoFOAM::parallelFor(
            exec(),
            ls.rhs().range(),
            KOKKOS_LAMBDA(const size_t i) { rhs[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> createEmptyLinearSystem() const
    {
        NeoFOAM::Field<NeoFOAM::scalar> values(exec(), 1, 0.0);
        NeoFOAM::Field<NeoFOAM::localIdx> colIdx(exec(), 1, 0.0);
        NeoFOAM::Field<NeoFOAM::localIdx> rowPtrs(exec(), {0, 1});
        NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> csrMatrix(
            values, colIdx, rowPtrs
        );

        NeoFOAM::Field<NeoFOAM::scalar> rhs(exec(), 1, 0.0);
        NeoFOAM::la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> linearSystem(
            csrMatrix, rhs, "diagonal"
        );
        return linearSystem;
    }

    std::string getName() const { return "TemporalDummy"; }
};

NeoFOAM::scalar getField(const NeoFOAM::Field<NeoFOAM::scalar>& source)
{
    auto sourceField = source.copyToHost();
    return sourceField.span()[0];
}


NeoFOAM::scalar getDiag(const la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.matrix().values()[0];
}

NeoFOAM::scalar getRhs(const la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.rhs().span()[0];
}
