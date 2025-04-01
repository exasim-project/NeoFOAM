// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors


#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/ddtOperator.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
DdtOperator<ValueType>::DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff(1.0), field, termType),
      sparsityPattern_(SparsityPattern::readOrCreate(field.mesh())) {};

template<typename ValueType>
void DdtOperator<ValueType>::explicitOperation(Field<ValueType>& source, scalar, scalar dt) const
{
    const scalar dtInver = 1.0 / dt;
    const auto vol = this->getField().mesh().cellVolumes().span();
    auto [sourceSpan, field, oldField] =
        spans(source, this->field_.internalField(), oldTime(this->field_).internalField());

    NeoFOAM::parallelFor(
        source.exec(),
        source.range(),
        KOKKOS_LAMBDA(const size_t celli) {
            sourceSpan[celli] += dtInver * (field[celli] - oldField[celli]) * vol[celli];
        }
    );
}

template<typename ValueType>
void DdtOperator<ValueType>::implicitOperation(
    la::LinearSystem<ValueType, localIdx>& ls, scalar, scalar dt
)
{
    const scalar dtInver = 1.0 / dt;
    const auto vol = this->getField().mesh().cellVolumes().span();
    const auto operatorScaling = this->getCoefficient();
    const auto [diagOffs, oldField] =
        spans(sparsityPattern_->diagOffset(), oldTime(this->field_).internalField());
    auto [A, b] = ls.view();

    NeoFOAM::parallelFor(
        ls.exec(),
        {0, oldField.size()},
        KOKKOS_LAMBDA(const size_t celli) {
            std::size_t idx = A.rowOffs[celli] + diagOffs[celli];
            const auto commonCoef = operatorScaling[celli] * vol[celli] * dtInver;
            A.values[idx] += commonCoef * one<ValueType>();
            b[celli] += commonCoef * oldField[celli];
        }
    );
}

// instantiate the template class
template class DdtOperator<scalar>;
template class DdtOperator<Vector>;

};
