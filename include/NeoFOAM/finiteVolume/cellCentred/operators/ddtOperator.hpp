// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/core/database/oldTimeCollection.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/sparsityPattern.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


template<typename ValueType>
class DdtOperator : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using FieldValueType = ValueType;

    DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field)
        : dsl::OperatorMixin<VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, termType
        ),
          sparsityPattern_(SparsityPattern::readOrCreate(field.mesh())) {

          };

    void explicitOperation(Field<ValueType>& source, scalar t, scalar dt)
    {
        const scalar rDeltat = 1 / dt;
        const auto vol = this->getField().mesh().cellVolumes().span();
        auto operatorScaling = this->getCoefficient();
        auto [sourceSpan, field, oldField] =
            spans(source, this->field_.internalField(), oldTime(this->field_).internalField());
        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t celli) {
                sourceSpan[celli] += rDeltat * (field[celli] - oldField[celli]) * vol[celli];
            }
        );
    }

    la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const
    {
        la::LinearSystem<scalar, localIdx> ls(sparsityPattern_->linearSystem());

        Field<ValueType> values(ls.matrix().exec(), ls.matrix().nValues(), zero<ValueType>());
        Field<localIdx> mColIdxs(
            ls.matrix().exec(), ls.matrix().colIdxs().data(), ls.matrix().nColIdxs()
        );
        Field<localIdx> mRowPtrs(
            ls.matrix().exec(), ls.matrix().rowPtrs().data(), ls.matrix().rowPtrs().size()
        );

        la::CSRMatrix<ValueType, localIdx> matrix(values, mColIdxs, mRowPtrs);
        Field<ValueType> rhs(ls.matrix().exec(), ls.rhs().size(), zero<ValueType>());

        return {matrix, rhs, ls.sparsityPattern()};
    }

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls, scalar t, scalar dt)
    {
        const scalar dtInver = 1.0 / dt;
        const auto vol = this->getField().mesh().cellVolumes().span();
        const auto operatorScaling = this->getCoefficient();
        const auto [diagOffs, oldField] =
            spans(sparsityPattern_->diagOffset(), oldTime(this->field_).internalField());
        const auto rowPtrs = ls.matrix().rowPtrs();
        const auto colIdxs = ls.matrix().colIdxs();
        auto values = ls.matrix().values();
        auto rhs = ls.rhs().span();

        NeoFOAM::parallelFor(
            ls.exec(),
            {0, oldField.size()},
            KOKKOS_LAMBDA(const size_t celli) {
                std::size_t idx = rowPtrs[celli] + diagOffs[celli];
                const auto commonCoef = operatorScaling[celli] * vol[celli] * dtInver; 
                values[idx] += commonCoef * one<ValueType>();
                rhs[celli] += commonCoef * oldField[celli];
            }
        );
    }


    void build(const Input& input)
    {
        // do nothing
    }

    std::string getName() const { return "DdtOperator"; }

private:

    // const VolumeField<ValueType> coefficients_;
    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};


} // namespace NeoFOAM
