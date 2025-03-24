// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

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

template<typename ValueType>
class Dummy : public NeoFOAM::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>
{

public:

    using FieldValueType = ValueType;

    Dummy(fvcc::VolumeField<ValueType>& field)
        : NeoFOAM::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, Operator::Type::Explicit
        )
    {}

    Dummy(fvcc::VolumeField<ValueType>& field, Operator::Type type)
        : NeoFOAM::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, type
        )
    {}

    void explicitOperation(NeoFOAM::Field<ValueType>& source)
    {
        auto sourceSpan = source.span();
        auto fieldSpan = this->field_.internalField().span();
        auto coeff = this->getCoefficient();
        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t i) { sourceSpan[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    void implicitOperation(la::LinearSystem<ValueType, NeoFOAM::localIdx>& ls)
    {
        auto values = ls.matrix().values().span();
        auto rhs = ls.rhs().span();
        auto fieldSpan = this->field_.internalField().span();
        auto coeff = this->getCoefficient();

        // update diag
        NeoFOAM::parallelFor(
            this->exec(),
            {0, values.size()},
            KOKKOS_LAMBDA(const size_t i) { values[i] += coeff[i] * fieldSpan[i]; }
        );

        // update rhs
        NeoFOAM::parallelFor(
            this->exec(),
            ls.rhs().range(),
            KOKKOS_LAMBDA(const size_t i) { rhs[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    la::LinearSystem<ValueType, NeoFOAM::localIdx> createEmptyLinearSystem() const
    {
        NeoFOAM::Field<ValueType> values(this->exec(), 1, NeoFOAM::zero<ValueType>());
        NeoFOAM::Field<NeoFOAM::localIdx> colIdx(this->exec(), 1, 0.0);
        NeoFOAM::Field<NeoFOAM::localIdx> rowPtrs(this->exec(), {0, 1});
        NeoFOAM::la::CSRMatrix<ValueType, NeoFOAM::localIdx> csrMatrix(values, colIdx, rowPtrs);

        NeoFOAM::Field<ValueType> rhs(this->exec(), 1, NeoFOAM::zero<ValueType>());
        NeoFOAM::la::LinearSystem<ValueType, NeoFOAM::localIdx> linearSystem(csrMatrix, rhs);
        return linearSystem;
    }

    std::string getName() const { return "Dummy"; }
};

/* A dummy implementation of a SpatialOperator
 * following the SpatialOperator interface */
template<typename ValueType>
class TemporalDummy : public NeoFOAM::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>
{

public:

    using FieldValueType = ValueType;

    TemporalDummy(fvcc::VolumeField<ValueType>& field)
        : NeoFOAM::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, Operator::Type::Explicit
        )
    {}

    TemporalDummy(fvcc::VolumeField<ValueType>& field, Operator::Type type)
        : NeoFOAM::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, type
        )
    {}

    void explicitOperation(NeoFOAM::Field<ValueType>& source, NeoFOAM::scalar t, NeoFOAM::scalar dt)
    {
        auto sourceSpan = source.span();
        auto fieldSpan = this->field_.internalField().span();
        auto coeff = this->getCoefficient();
        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t i) { sourceSpan[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    void implicitOperation(
        la::LinearSystem<ValueType, NeoFOAM::localIdx>& ls, NeoFOAM::scalar t, NeoFOAM::scalar dt
    )
    {
        auto values = ls.matrix().values().span();
        auto rhs = ls.rhs().span();
        auto fieldSpan = this->field_.internalField().span();
        auto coeff = this->getCoefficient();

        // update diag
        NeoFOAM::parallelFor(
            this->exec(),
            {0, values.size()},
            KOKKOS_LAMBDA(const size_t i) { values[i] += coeff[i] * fieldSpan[i]; }
        );

        // update rhs
        NeoFOAM::parallelFor(
            this->exec(),
            ls.rhs().range(),
            KOKKOS_LAMBDA(const size_t i) { rhs[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    la::LinearSystem<ValueType, NeoFOAM::localIdx> createEmptyLinearSystem() const
    {
        NeoFOAM::Field<ValueType> values(this->exec(), 1, NeoFOAM::zero<ValueType>());
        NeoFOAM::Field<NeoFOAM::localIdx> colIdx(this->exec(), 1, 0.0);
        NeoFOAM::Field<NeoFOAM::localIdx> rowPtrs(this->exec(), {0, 1});
        NeoFOAM::la::CSRMatrix<ValueType, NeoFOAM::localIdx> csrMatrix(values, colIdx, rowPtrs);

        NeoFOAM::Field<ValueType> rhs(this->exec(), 1, NeoFOAM::zero<ValueType>());
        NeoFOAM::la::LinearSystem<ValueType, NeoFOAM::localIdx> linearSystem(csrMatrix, rhs);
        return linearSystem;
    }

    std::string getName() const { return "TemporalDummy"; }
};

template<typename ValueType>
ValueType getField(const NeoFOAM::Field<ValueType>& source)
{
    auto sourceField = source.copyToHost();
    return sourceField.span()[0];
}

template<typename ValueType>
ValueType getDiag(const la::LinearSystem<ValueType, NeoFOAM::localIdx> ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.matrix().values()[0];
}

template<typename ValueType>
ValueType getRhs(const la::LinearSystem<ValueType, NeoFOAM::localIdx> ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.rhs().span()[0];
}
