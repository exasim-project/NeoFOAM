// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/operator.hpp"
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
          sparsityPattern_(SparsityPattern::readOrCreate(field.mesh())) {};

    void explicitOperation(Field<scalar>& source)
    {
        auto operatorScaling = getCoefficient();
        auto [sourceSpan, field, oldField] =
            spans(source, field_.internalField(), oldTime(field_).internalField());
        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t celli) {
                sourceSpan[celli] += field[celli] - oldField[celli];
            }
        );
    }

    la::LinearSystem<scalar, localIdx> createEmptyLinearSystem() const
    {
        return sparsityPattern_->linearSystem();
    }

    void implicitOperation(la::LinearSystem<scalar, localIdx>& ls)
    {
        const auto operatorScaling = getCoefficient();
        const auto [diagOffs, oldField] =
            spans(sparsityPattern_->diagOffset(), oldTime(field_).internalField());
        const auto rowPtrs = ls.matrix().rowPtrs();
        const auto colIdxs = ls.matrix().colIdxs();
        std::span<scalar> values = ls.matrix().values();
        NeoFOAM::parallelFor(
            ls.exec(),
            {0, coeff.size()},
            KOKKOS_LAMBDA(const size_t celli) {
                std::size_t idx = rowPtrs[celli] + diagOffs[celli];
            }
        );
    }


    void build(const Input& input)
    {
        // do nothing
    }

    std::string getName() const { return "DivOperator"; }

private:

    const VolumeField<scalar>& coefficients_;
    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};


} // namespace NeoFOAM
