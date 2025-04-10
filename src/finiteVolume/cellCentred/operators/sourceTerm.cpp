// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors


#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/sourceTerm.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
SourceTerm<ValueType>::SourceTerm(
    dsl::Operator::Type termType, VolumeField<scalar>& coefficients, VolumeField<ValueType>& field
)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff(1.0), field, termType),
      coefficients_(coefficients), sparsityPattern_(SparsityPattern::readOrCreate(field.mesh())) {};

template<typename ValueType>
void SourceTerm<ValueType>::explicitOperation(Field<ValueType>& source) const
{
    auto operatorScaling = this->getCoefficient();
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

template<typename ValueType>
void SourceTerm<ValueType>::implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) const
{
    const auto operatorScaling = this->getCoefficient();
    const auto vol = coefficients_.mesh().cellVolumes().view();
    const auto [diagOffs, coeff] =
        spans(sparsityPattern_->diagOffset(), coefficients_.internalField());
    auto [matrix, rhs] = ls.view();

    NeoFOAM::parallelFor(
        ls.exec(),
        {0, coeff.size()},
        KOKKOS_LAMBDA(const size_t celli) {
            std::size_t idx = matrix.rowOffs[celli] + diagOffs[celli];
            matrix.values[idx] +=
                operatorScaling[celli] * coeff[celli] * vol[celli] * one<ValueType>();
        }
    );
}


// instantiate the template class
template class SourceTerm<scalar>;
template class SourceTerm<Vector>;
};
