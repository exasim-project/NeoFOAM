// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

using Field = NeoN::Field<NeoN::scalar>;
using Coeff = NeoN::dsl::Coeff;
using Operator = NeoN::dsl::Operator;
using Executor = NeoN::Executor;
using VolumeField = fvcc::VolumeField<NeoN::scalar>;
using OperatorMixin = NeoN::dsl::OperatorMixin<VolumeField>;
using BoundaryFields = NeoN::BoundaryFields<NeoN::scalar>;

/* A dummy implementation of a SpatialOperator
 * following the SpatialOperator interface */

template<typename ValueType>
class Dummy : public NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>
{

public:

    using FieldValueType = ValueType;

    Dummy(fvcc::VolumeField<ValueType>& field)
        : NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, Operator::Type::Explicit
        )
    {}

    Dummy(fvcc::VolumeField<ValueType>& field, Operator::Type type)
        : NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, type
        )
    {}

    void explicitOperation(NeoN::Field<ValueType>& source)
    {
        auto sourceSpan = source.span();
        auto fieldSpan = this->field_.internalField().span();
        auto coeff = this->getCoefficient();
        NeoN::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t i) { sourceSpan[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    void implicitOperation(la::LinearSystem<ValueType, NeoN::localIdx>& ls)
    {
        auto values = ls.matrix().values().span();
        auto rhs = ls.rhs().span();
        auto fieldSpan = this->field_.internalField().span();
        auto coeff = this->getCoefficient();

        // update diag
        NeoN::parallelFor(
            this->exec(),
            {0, values.size()},
            KOKKOS_LAMBDA(const size_t i) { values[i] += coeff[i] * fieldSpan[i]; }
        );

        // update rhs
        NeoN::parallelFor(
            this->exec(),
            ls.rhs().range(),
            KOKKOS_LAMBDA(const size_t i) { rhs[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    la::LinearSystem<ValueType, NeoN::localIdx> createEmptyLinearSystem() const
    {
        NeoN::Field<ValueType> values(this->exec(), 1, NeoN::zero<ValueType>());
        NeoN::Field<NeoN::localIdx> colIdx(this->exec(), 1, 0.0);
        NeoN::Field<NeoN::localIdx> rowPtrs(this->exec(), {0, 1});
        NeoN::la::CSRMatrix<ValueType, NeoN::localIdx> csrMatrix(values, colIdx, rowPtrs);

        NeoN::Field<ValueType> rhs(this->exec(), 1, NeoN::zero<ValueType>());
        NeoN::la::LinearSystem<ValueType, NeoN::localIdx> linearSystem(csrMatrix, rhs);
        return linearSystem;
    }

    std::string getName() const { return "Dummy"; }
};

/* A dummy implementation of a SpatialOperator
 * following the SpatialOperator interface */
template<typename ValueType>
class TemporalDummy : public NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>
{

public:

    using FieldValueType = ValueType;

    TemporalDummy(fvcc::VolumeField<ValueType>& field)
        : NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, Operator::Type::Explicit
        )
    {}

    TemporalDummy(fvcc::VolumeField<ValueType>& field, Operator::Type type)
        : NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, type
        )
    {}

    void explicitOperation(NeoN::Field<ValueType>& source, NeoN::scalar, NeoFOAM::scalar)
    {
        auto sourceSpan = source.span();
        auto fieldSpan = this->field_.internalField().span();
        auto coeff = this->getCoefficient();
        NeoN::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t i) { sourceSpan[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    void implicitOperation(
        la::LinearSystem<ValueType, NeoN::localIdx>& ls, NeoN::scalar, NeoFOAM::scalar
    )
    {
        auto values = ls.matrix().values().span();
        auto rhs = ls.rhs().span();
        auto fieldSpan = this->field_.internalField().span();
        auto coeff = this->getCoefficient();

        // update diag
        NeoN::parallelFor(
            this->exec(),
            {0, values.size()},
            KOKKOS_LAMBDA(const size_t i) { values[i] += coeff[i] * fieldSpan[i]; }
        );

        // update rhs
        NeoN::parallelFor(
            this->exec(),
            ls.rhs().range(),
            KOKKOS_LAMBDA(const size_t i) { rhs[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    la::LinearSystem<ValueType, NeoN::localIdx> createEmptyLinearSystem() const
    {
        NeoN::Field<ValueType> values(this->exec(), 1, NeoN::zero<ValueType>());
        NeoN::Field<NeoN::localIdx> colIdx(this->exec(), 1, 0.0);
        NeoN::Field<NeoN::localIdx> rowPtrs(this->exec(), {0, 1});
        NeoN::la::CSRMatrix<ValueType, NeoN::localIdx> csrMatrix(values, colIdx, rowPtrs);

        NeoN::Field<ValueType> rhs(this->exec(), 1, NeoN::zero<ValueType>());
        NeoN::la::LinearSystem<ValueType, NeoN::localIdx> linearSystem(csrMatrix, rhs);
        return linearSystem;
    }

    std::string getName() const { return "TemporalDummy"; }
};

template<typename ValueType>
ValueType getField(const NeoN::Field<ValueType>& source)
{
    auto sourceField = source.copyToHost();
    return sourceField.span()[0];
}

template<typename ValueType>
ValueType getDiag(const la::LinearSystem<ValueType, NeoN::localIdx>& ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.matrix().values().span()[0];
}

template<typename ValueType>
ValueType getRhs(const la::LinearSystem<ValueType, NeoN::localIdx>& ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.rhs().span()[0];
}
