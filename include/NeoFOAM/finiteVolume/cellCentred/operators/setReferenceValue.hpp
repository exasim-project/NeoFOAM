// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/sparsityPattern.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


template<typename ValueType>
class SetReferenceValue : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    SetReferenceValue(
        VolumeField<ValueType>& field, const ValueType refValue, const std::size_t refCell
    )
        : dsl::OperatorMixin<VolumeField<scalar>>(
            field.exec(), dsl::Coeff(1.0), field, dsl::Operator::Type::Implicit
        ),
          refValue_(refValue), refCell_(refCell),
          sparsityPattern_(SparsityPattern::readOrCreate(field.mesh())) {};


    la::LinearSystem<scalar, localIdx> createEmptyLinearSystem() const
    {
        return sparsityPattern_->linearSystem();
    }

    void implicitOperation(la::LinearSystem<scalar, localIdx>& ls)
    {
        const auto diagOffset = sparsityPattern_->diagOffset().span();
        const auto rowPtrs = ls.matrix().rowPtrs();
        auto rhs = ls.rhs().span();
        auto values = ls.matrix().values();
        const auto refValue = refValue_;
        NeoFOAM::parallelFor(
            ls.exec(),
            {refCell_, refCell_ + 1},
            KOKKOS_LAMBDA(const std::size_t refCelli) {
                auto diagIdx = rowPtrs[refCelli] + diagOffset[refCelli];
                auto diagValue = values[diagIdx];
                rhs[refCelli] += diagValue * refValue;
                values[diagIdx] += diagValue;
            }
        );
    }


    void build(const Input& input)
    {
        // do nothing
    }

    std::string getName() const { return "DivOperator"; }

private:

    const ValueType refValue_;
    const std::size_t refCell_;
    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};


} // namespace NeoFOAM
