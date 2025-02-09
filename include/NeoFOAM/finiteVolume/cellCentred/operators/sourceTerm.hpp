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
class SourceTerm : public dsl::OperatorMixin<VolumeField<scalar>>
{

public:

    SourceTerm(
        dsl::Operator::Type termType, VolumeField<scalar>& coefficients, VolumeField<scalar>& field
    )
        : dsl::OperatorMixin<VolumeField<scalar>>(field.exec(), field, termType),
          coefficients_(coefficients),
          sparsityPattern_(SparsityPattern::readOrCreate(field.mesh())) {};

    void explicitOperation(Field<scalar>& source)
    {
        auto operatorScaling = getCoefficient();
        const auto vol = coefficients_.mesh().cellVolumes().span();
        auto [sourceSpan, fieldSpan, coeff] =
            spans(source, field_.internalField(), coefficients_.internalField());
        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t celli) {
                sourceSpan[celli] += operatorScaling[celli] * coeff[celli] * fieldSpan[celli];
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
        const auto vol = coefficients_.mesh().cellVolumes().span();
        const auto [diagOffs, coeff] =
            spans(sparsityPattern_->diagOffset(), coefficients_.internalField());
        const auto rowPtrs = ls.matrix().rowPtrs();
        const auto colIdxs = ls.matrix().colIdxs();
        std::span<scalar> values = ls.matrix().values();
        NeoFOAM::parallelFor(
            ls.exec(),
            {0, coeff.size()},
            KOKKOS_LAMBDA(const size_t celli) {
                std::size_t idx = rowPtrs[celli] + diagOffs[celli];
                values[idx] += operatorScaling[celli] * coeff[celli] * vol[celli];
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
