// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/fvccSparsityPattern.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


// template<typename ValueType>
class DdtOperator : public dsl::OperatorMixin<VolumeField<scalar>>
{

public:

    DdtOperator(dsl::Operator::Type termType, VolumeField<scalar>& field)
        : dsl::OperatorMixin<VolumeField<scalar>>(field.exec(), field, termType),
          sparsityPattern_(SparsityPattern::readOrCreate(field.mesh())) {

          };

    void explicitOperation(Field<scalar>& source, scalar t, scalar dt)
    {
        const scalar rDeltat = 1 / dt;
        const auto vol = getField().mesh().cellVolumes().span();
        auto operatorScaling = getCoefficient();
        auto [sourceSpan, field, oldField] =
            spans(source, field_.internalField(), oldTime(field_).internalField());
        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t celli) {
                sourceSpan[celli] += rDeltat * (field[celli] - oldField[celli]) * vol[celli];
            }
        );
    }

    la::LinearSystem<scalar, localIdx> createEmptyLinearSystem() const
    {
        return sparsityPattern_->linearSystem();
    }

    void implicitOperation(la::LinearSystem<scalar, localIdx>& ls, scalar t, scalar dt)
    {
        const scalar rDeltat = 1 / dt;
        const auto vol = getField().mesh().cellVolumes().span();
        const auto operatorScaling = getCoefficient();
        const auto [diagOffs, oldField] =
            spans(sparsityPattern_->diagOffset(), oldTime(field_).internalField());
        const auto rowPtrs = ls.matrix().rowPtrs();
        const auto colIdxs = ls.matrix().colIdxs();
        std::span<scalar> values = ls.matrix().values();
        std::span<scalar> rhs = ls.rhs().span();
        NeoFOAM::parallelFor(
            ls.exec(),
            {0, oldField.size()},
            KOKKOS_LAMBDA(const size_t celli) {
                std::size_t idx = rowPtrs[celli] + diagOffs[celli];
                values[idx] += operatorScaling[celli] * rDeltat * vol[celli];
                rhs[celli] += operatorScaling[celli] * rDeltat * oldField[celli] * vol[celli];
            }
        );
    }


    void build(const Input& input)
    {
        // do nothing
    }

    std::string getName() const { return "DdtOperator"; }

private:

    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};


} // namespace NeoFOAM
