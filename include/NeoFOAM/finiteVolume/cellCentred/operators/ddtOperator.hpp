// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/core/database/oldTimeCollection.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"

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
          sparsityPattern_(SparsityPattern::readOrCreate(field.mesh())) {};

    void explicitOperation(Field<ValueType>& source, scalar, scalar dt) const
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

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls, scalar, scalar dt)
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
                std::size_t idx = A.rowOffset[celli] + diagOffs[celli];
                const auto commonCoef = operatorScaling[celli] * vol[celli] * dtInver;
                A.value[idx] += commonCoef * one<ValueType>();
                b[celli] += commonCoef * oldField[celli];
            }
        );
    }

    void build([[maybe_unused]] const Input& input)
    {
        // do nothing
    }

    std::string getName() const { return "DdtOperator"; }

private:

    // const VolumeField<ValueType> coefficients_;
    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};


} // namespace NeoFOAM
