// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/field.hpp"


namespace NeoFOAM::la
{

template<typename ValueType, typename IndexType>
class CSRMatrix
{

public:

    CSRMatrix(
        const Field<ValueType>& values,
        const Field<IndexType>& colIdxs,
        const Field<IndexType>& rowPtrs
    )
        : values_(values), colIdxs_(colIdxs), rowPtrs_(rowPtrs)
    {
        NF_ASSERT(values.exec() == colIdxs.exec(), "Executors are not the same");
        NF_ASSERT(values.exec() == rowPtrs.exec(), "Executors are not the same");
    };

    ~CSRMatrix() = default;

    [[nodiscard]] const Executor& exec() const { return values_.exec(); }

    [[nodiscard]] IndexType nRows() const { return rowPtrs_.size() - 1; }

    [[nodiscard]] IndexType nValues() const { return values_.size(); }

    [[nodiscard]] IndexType nColIdxs() const { return colIdxs_.size(); }

    [[nodiscard]] std::span<ValueType> values() { return values_.span(); }
    [[nodiscard]] std::span<IndexType> colIdxs() { return colIdxs_.span(); }
    [[nodiscard]] std::span<IndexType> rowPtrs() { return rowPtrs_.span(); }

    [[nodiscard]] const std::span<const ValueType> values() const { return values_.span(); }
    [[nodiscard]] const std::span<const IndexType> colIdxs() const { return colIdxs_.span(); }
    [[nodiscard]] const std::span<const IndexType> rowPtrs() const { return rowPtrs_.span(); }


private:

    Field<ValueType> values_;
    Field<IndexType> colIdxs_;
    Field<IndexType> rowPtrs_;
};

} // namespace NeoFOAM
