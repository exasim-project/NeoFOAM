// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


template<typename ValueType>
class SourceTerm : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using FieldValueType = ValueType;

    SourceTerm(
        dsl::Operator::Type termType,
        VolumeField<scalar>& coefficients,
        VolumeField<ValueType>& field
    )
        : dsl::OperatorMixin<VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, termType
        ),
          coefficients_(coefficients),
          sparsityPattern_(SparsityPattern::readOrCreate(field.mesh())) {};

    void explicitOperation(Field<ValueType>& source) const
    {
        auto operatorScaling = this->getCoefficient();
        const auto vol = coefficients_.mesh().cellVolumes().span();
        auto [sourceSpan, fieldSpan, coeff] =
            spans(source, this->field_.internalField(), coefficients_.internalField());
        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t celli) {
                sourceSpan[celli] += operatorScaling[celli] * coeff[celli] * fieldSpan[celli];
            }
        );
    }

    // FIXME: remove?
    la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const
    {
        la::LinearSystem<scalar, localIdx> ls(sparsityPattern_->linearSystem());
        auto [A, b] = ls.view();
        const auto& exec = ls.exec();

        Field<ValueType> values(exec, A.value.size(), zero<ValueType>());
        Field<localIdx> mColIdxs(exec, A.columnIndex.data(), A.columnIndex.size());
        Field<localIdx> mRowPtrs(exec, A.rowOffset.data(), A.rowOffset.size());

        la::CSRMatrix<ValueType, localIdx> matrix(values, mColIdxs, mRowPtrs);
        Field<ValueType> rhs(exec, b.size(), zero<ValueType>());

        return {matrix, rhs};
    }

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls)
    {
        const auto operatorScaling = this->getCoefficient();
        const auto vol = coefficients_.mesh().cellVolumes().span();
        const auto [diagOffs, coeff] =
            spans(sparsityPattern_->diagOffset(), coefficients_.internalField());
        auto [A, b] = ls.view();

        NeoFOAM::parallelFor(
            ls.exec(),
            {0, coeff.size()},
            KOKKOS_LAMBDA(const size_t celli) {
                std::size_t idx = A.rowOffset[celli] + diagOffs[celli];
                A.value[idx] +=
                    operatorScaling[celli] * coeff[celli] * vol[celli] * one<ValueType>();
            }
        );
    }


    void build([[maybe_unused]] const Input& input)
    {
        // do nothing
    }

    std::string getName() const { return "DivOperator"; }

private:

    const VolumeField<scalar>& coefficients_;
    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};


} // namespace NeoFOAM
