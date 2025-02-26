// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/field.hpp"


namespace NeoFOAM::la
{

enum BlockStructType
{
    cell,
    component
};

template<typename ValueType, typename IndexType>
class CSRMatrixSpan
{
public:

    // Constructor for non-const spans
    CSRMatrixSpan(
        const std::span<ValueType>& values,
        const std::span<IndexType>& colIdxs,
        const std::span<IndexType>& rowPtrs
    )
        : values_(values), colIdxs_(colIdxs), rowPtrs_(rowPtrs) {};

    // Constructor for const spans - needed for const matrix objects
    CSRMatrixSpan(
        const std::span<const ValueType>& values,
        const std::span<const IndexType>& colIdxs,
        const std::span<const IndexType>& rowPtrs
    )
        : values_(values), colIdxs_(colIdxs), rowPtrs_(rowPtrs) {};

    ~CSRMatrixSpan() = default;

    KOKKOS_INLINE_FUNCTION
    ValueType& entry(const IndexType i, const IndexType j) const
    {
        for (IndexType ic = 0; ic < (rowPtrs_[i + 1] - rowPtrs_[i]); ++ic)
        {
            if (colIdxs_[rowPtrs_[i] + ic] == j)
            {
                return values_[rowPtrs_[i] + ic];
            }
            if (colIdxs_[rowPtrs_[i] + ic] > j) break;
        }
        Kokkos::abort("Memory not allocated for CSR matrix component.");
    }

private:

    std::span<ValueType> values_;
    std::span<IndexType> colIdxs_;
    std::span<IndexType> rowPtrs_;
};

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

    [[nodiscard]] CSRMatrix<ValueType, IndexType> copyToExecutor(Executor dstExec) const
    {
        if (dstExec == values_.exec()) return *this;
        CSRMatrix<ValueType, IndexType> result(
            values_.copyToHost(), colIdxs_.copyToHost(), rowPtrs_.copyToHost()
        );
        result.type_ = type_;
        return result;
    }

    [[nodiscard]] CSRMatrix<ValueType, IndexType> copyToHost() const
    {
        return copyToExecutor(SerialExecutor());
    }

    [[nodiscard]] CSRMatrixSpan<ValueType, IndexType> span()
    {
        return CSRMatrixSpan(values_.span(), colIdxs_.span(), rowPtrs_.span());
    }

    [[nodiscard]] const CSRMatrixSpan<const ValueType, const IndexType> span() const
    {
        return CSRMatrixSpan(values_.span(), colIdxs_.span(), rowPtrs_.span());
    }

private:

    BlockStructType type_ {cell};
    Field<ValueType> values_;
    Field<IndexType> colIdxs_;
    Field<IndexType> rowPtrs_;
};

} // namespace NeoFOAM
