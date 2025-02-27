// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/field.hpp"


namespace NeoFOAM::la
{


/**
 * @class CSRMatrixSpan
 * @brief A helper class to allow easy read/write on all executors.
 * @tparam ValueType The type of the underlying CSR matrix elements.
 * @tparam IndexType The type of the indexes.
 */
template<typename ValueType, typename IndexType>
class CSRMatrixSpan
{
public:

    /**
     * @brief Constructor for CSRMatrixSpan.
     * @param values Span of the non-zero values of the matrix.
     * @param colIdxs Span of the column indices for each non-zero value.
     * @param rowPtrs Span of the starting index in values/colIdxs for each row.
     */
    CSRMatrixSpan(
        const std::span<ValueType>& values,
        const std::span<IndexType>& colIdxs,
        const std::span<IndexType>& rowPtrs
    )
        : values_(values), colIdxs_(colIdxs), rowPtrs_(rowPtrs) {};

    /**
     * @brief Default destructor.
     */
    ~CSRMatrixSpan() = default;

    /**
     * @brief Retrieve a reference to the matrix element at position (i,j).
     * @param i The row index.
     * @param j The column index.
     * @return Reference to the matrix element if it exists.
     */
    KOKKOS_INLINE_FUNCTION
    ValueType& entry(const IndexType i, const IndexType j) const
    {
        const IndexType rowSize = rowPtrs_[i + 1] - rowPtrs_[i];
        for (std::remove_const_t<IndexType> ic = 0; ic < rowSize; ++ic)
        {
            const IndexType localCol = rowPtrs_[i] + ic;
            if (colIdxs_[localCol] == j)
            {
                return values_[localCol];
            }
            if (colIdxs_[localCol] > j) break;
        }
        Kokkos::abort("Memory not allocated for CSR matrix component.");
        return values_[values_.size()]; // compiler warning suppression.
    }

    /**
     * @brief Direct access to a value given the offset.
     * @param offset The offset, from 0, to the value.
     * @return Reference to the matrix element if it exists.
     */
    KOKKOS_INLINE_FUNCTION
    ValueType& directValue(const IndexType offset) const { values_[offset]; }

private:

    std::span<ValueType> values_;  //!< Span to the values of the CSR matrix.
    std::span<IndexType> colIdxs_; //!< Span to the column indices of the CSR matrix.
    std::span<IndexType> rowPtrs_; //!< Span to the row offsets for the CSR matrix.
};

template<typename ValueType, typename IndexType>
class CSRMatrix
{

public:

    /**
     * @brief Constructor for CSRMatrix.
     * @param values The non-zero values of the matrix.
     * @param colIdxs The column indices for each non-zero value.
     * @param rowPtrs The starting index in values/colIdxs for each row.
     */
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

    /**
     * @brief Default destructor.
     */
    ~CSRMatrix() = default;

    /**
     * @brief Get the executor associated with this matrix.
     * @return Reference to the executor.
     */
    [[nodiscard]] const Executor& exec() const { return values_.exec(); }

    /**
     * @brief Get the number of rows in the matrix.
     * @return Number of rows.
     */
    [[nodiscard]] IndexType nRows() const { return rowPtrs_.size() - 1; }

    /**
     * @brief Get the number of non-zero values in the matrix.
     * @return Number of non-zero values.
     */
    [[nodiscard]] IndexType nValues() const { return values_.size(); }

    /**
     * @brief Get the number of column indices in the matrix.
     * @return Number of column indices.
     */
    [[nodiscard]] IndexType nColIdxs() const { return colIdxs_.size(); }

    /**
     * @brief Get a span to the values array.
     * @return Span containing the matrix values.
     */
    [[nodiscard]] std::span<ValueType> values() { return values_.span(); }

    /**
     * @brief Get a span to the column indices array.
     * @return Span containing the column indices.
     */
    [[nodiscard]] std::span<IndexType> colIdxs() { return colIdxs_.span(); }

    /**
     * @brief Get a span to the row pointers array.
     * @return Span containing the row pointers.
     */
    [[nodiscard]] std::span<IndexType> rowPtrs() { return rowPtrs_.span(); }

    /**
     * @brief Get a const span to the values array.
     * @return Const span containing the matrix values.
     */
    [[nodiscard]] const std::span<const ValueType> values() const { return values_.span(); }

    /**
     * @brief Get a const span to the column indices array.
     * @return Const span containing the column indices.
     */
    [[nodiscard]] const std::span<const IndexType> colIdxs() const { return colIdxs_.span(); }

    /**
     * @brief Get a const span to the row pointers array.
     * @return Const span containing the row pointers.
     */
    [[nodiscard]] const std::span<const IndexType> rowPtrs() const { return rowPtrs_.span(); }

    /**
     * @brief Copy the matrix to another executor.
     * @param dstExec The destination executor.
     * @return A copy of the matrix on the destination executor.
     */
    [[nodiscard]] CSRMatrix<ValueType, IndexType> copyToExecutor(Executor dstExec) const
    {
        if (dstExec == values_.exec()) return *this;
        CSRMatrix<ValueType, IndexType> other(
            values_.copyToHost(), colIdxs_.copyToHost(), rowPtrs_.copyToHost()
        );
        return other;
    }

    /**
     * @brief Copy the matrix to the host.
     * @return A copy of the matrix on the host.
     */
    [[nodiscard]] CSRMatrix<ValueType, IndexType> copyToHost() const
    {
        return copyToExecutor(SerialExecutor());
    }

    /**
     * @brief Get a span representation of the matrix.
     * @return CSRMatrixSpan for easy access to matrix elements.
     */
    [[nodiscard]] CSRMatrixSpan<ValueType, IndexType> span()
    {
        return CSRMatrixSpan(values_.span(), colIdxs_.span(), rowPtrs_.span());
    }

    /**
     * @brief Get a const span representation of the matrix.
     * @return Const CSRMatrixSpan for read-only access to matrix elements.
     */
    [[nodiscard]] const CSRMatrixSpan<const ValueType, const IndexType> span() const
    {
        return CSRMatrixSpan(values_.span(), colIdxs_.span(), rowPtrs_.span());
    }

private:

    Field<ValueType> values_;  //!< The (non-zero) values of the CSR matrix.
    Field<IndexType> colIdxs_; //!< The column indices of the CSR matrix.
    Field<IndexType> rowPtrs_; //!< The row offsets for the CSR matrix.
};

} // namespace NeoFOAM
