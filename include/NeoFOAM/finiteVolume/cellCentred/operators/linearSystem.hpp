// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoFOAM/linearAlgebra/CSRMatrix.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/linearAlgebra/ginkgo.hpp"

#include "NeoFOAM/finiteVolume/cellCentred/operators/sparsityPattern.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType, typename IndexType = localIdx>
class LinearSystem
{
public:

    LinearSystem(VolumeField<ValueType>& psi)
        : psi_(psi), ls_(SparsityPattern::readOrCreate(psi.mesh())->linearSystem()),
          sparsityPattern_(SparsityPattern::readOrCreate(psi.mesh())) {

          };

    LinearSystem(
        VolumeField<ValueType>& psi,
        const la::LinearSystem<ValueType, IndexType>& ls,
        std::shared_ptr<SparsityPattern> sparsityPattern
    )
        : psi_(psi), ls_(ls), sparsityPattern_(sparsityPattern) {};

    LinearSystem(const LinearSystem& ls)
        : psi_(ls.psi_), ls_(ls.ls_), sparsityPattern_(ls.sparsityPattern_) {};

    ~LinearSystem() = default;

    [[nodiscard]] la::LinearSystem<ValueType, IndexType>& linearSystem() { return ls_; }
    [[nodiscard]] SparsityPattern& sparsityPattern()
    {
        if (!sparsityPattern_)
        {
            NF_THROW("fvcc:LinearSystem:sparsityPattern: sparsityPattern is null");
        }
        return *sparsityPattern_;
    }

    [[nodiscard]] const la::LinearSystem<ValueType, IndexType>& linearSystem() const { return ls_; }
    [[nodiscard]] const SparsityPattern& sparsityPattern() const
    {
        if (!sparsityPattern_)
        {
            NF_THROW("fvcc:LinearSystem:sparsityPattern: sparsityPattern is null");
        }
        return *sparsityPattern_;
    }

    const Executor& exec() const { return ls_.exec(); }

    void diag(Field<ValueType>& field)
    {
        NF_ASSERT_EQUAL(field.size(), sparsityPattern_->diagOffset().size());
        const auto diagOffset = sparsityPattern_->diagOffset().span();
        const auto rowPtrs = ls_.matrix().rowPtrs();
        std::span<ValueType> fieldSpan = field.span();
        std::span<ValueType> values = ls_.matrix().values();
        NeoFOAM::parallelFor(
            exec(),
            {0, diagOffset.size()},
            KOKKOS_LAMBDA(const std::size_t celli) {
                auto diagOffsetCelli = diagOffset[celli];
                fieldSpan[celli] = values[rowPtrs[celli] + diagOffsetCelli];
            }
        );
    }

    Field<IndexType> diagIndex()
    {
        Field<IndexType> diagIndex(exec(), sparsityPattern_->diagOffset().size());
        const auto diagOffset = sparsityPattern_->diagOffset().span();
        auto diagIndexSpan = diagIndex.span();
        const auto rowPtrs = ls_.matrix().rowPtrs();
        NeoFOAM::parallelFor(
            exec(),
            {0, diagIndex.size()},
            KOKKOS_LAMBDA(const std::size_t celli) {
                diagIndexSpan[celli] = rowPtrs[celli] + diagOffset[celli];
            }
        );
        return diagIndex;
    }

private:

    VolumeField<ValueType>& psi_;
    la::LinearSystem<ValueType, IndexType> ls_;
    std::shared_ptr<SparsityPattern> sparsityPattern_;
};

template<typename ValueType, typename IndexType = localIdx>
VolumeField<ValueType>
operator&(const LinearSystem<ValueType, IndexType> ls, const VolumeField<ValueType>& psi)
{
    VolumeField<ValueType> resultField(
        psi.exec(),
        "ls_" + psi.name,
        psi.mesh(),
        psi.internalField(),
        psi.boundaryField(),
        psi.boundaryConditions()
    );

    auto [result, b, x] =
        spans(resultField.internalField(), ls.linearSystem().rhs(), psi.internalField());

    const std::span<const ValueType> values = ls.linearSystem().matrix().values();
    const std::span<const IndexType> colIdxs = ls.linearSystem().matrix().colIdxs();
    const std::span<const IndexType> rowPtrs = ls.linearSystem().matrix().rowPtrs();

    NeoFOAM::parallelFor(
        resultField.exec(),
        {0, result.size()},
        KOKKOS_LAMBDA(const std::size_t rowi) {
            IndexType rowStart = rowPtrs[rowi];
            IndexType rowEnd = rowPtrs[rowi + 1];
            ValueType sum = 0.0;
            for (IndexType coli = rowStart; coli < rowEnd; coli++)
            {
                sum += values[coli] * x[colIdxs[coli]];
            }
            result[rowi] = sum - b[rowi];
        }
    );

    return resultField;
}

}
