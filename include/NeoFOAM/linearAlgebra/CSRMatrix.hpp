// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <tuple>

#include "NeoFOAM/fields/field.hpp"


namespace NeoFOAM::la
{

/**
 * @class CSRMatrixView
 * @brief A helper class to allow easy read/write on all executors.
 * @tparam ValueType The type of the underlying CSR matrix elements.
 * @tparam IndexType The type of the indexes.
 */
template<typename ValueType, typename IndexType>
class CSRMatrixView
{
public:

    /**
     * @brief Constructor for CSRMatrixView.
     * @param values Span of the non-zero values of the matrix.
     * @param colIdxs Span of the column indices for each non-zero value.
     * @param rowPtrs Span of the starting index in values/colIdxs for each row.
     */
    CSRMatrixView(
        const std::span<ValueType>& inValue,
        const std::span<IndexType>& inColumnIndex,
        const std::span<IndexType>& inRwOffset
    )
        : value(inValue), columnIndex(inColumnIndex), rowOffset(inRwOffset) {};

    /**
     * @brief Default destructor.
     */
    ~CSRMatrixView() = default;

    /**
     * @brief Retrieve a reference to the matrix element at position (i,j).
     * @param i The row index.
     * @param j The column index.
     * @return Reference to the matrix element if it exists.
     */
    KOKKOS_INLINE_FUNCTION
    ValueType& entry(const IndexType i, const IndexType j) const
    {
        const IndexType rowSize = rowOffset[i + 1] - rowOffset[i];
        for (std::remove_const_t<IndexType> ic = 0; ic < rowSize; ++ic)
        {
            const IndexType localCol = rowOffset[i] + ic;
            if (columnIndex[localCol] == j)
            {
                return value[localCol];
            }
            if (columnIndex[localCol] > j) break;
        }
        Kokkos::abort("Memory not allocated for CSR matrix component.");
        return value[value.size()]; // compiler warning suppression.
    }

    /**
     * @brief Direct access to a value given the offset.
     * @param offset The offset, from 0, to the value.
     * @return Reference to the matrix element if it exists.
     */
    KOKKOS_INLINE_FUNCTION
    ValueType& entry(const IndexType offset) const { return value[offset]; }

    std::span<ValueType> value;       //!< Span to the values of the CSR matrix.
    std::span<IndexType> columnIndex; //!< Span to the column indices of the CSR matrix.
    std::span<IndexType> rowOffset;   //!< Span to the row offsets for the CSR matrix.
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

    CSRMatrix(const Executor exec) : values_(exec, 0), colIdxs_(exec, 0), rowPtrs_(exec, 0) {}

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
    [[nodiscard]] IndexType nRows() const { return static_cast<IndexType>(rowPtrs_.size()) - 1; }

    /**
     * @brief Get the number of non-zero values in the matrix.
     * @return Number of non-zero values.
     */
    [[nodiscard]] IndexType nNonZeros() const { return static_cast<IndexType>(values_.size()); }

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
        if (dstExec == values_.exec())
        {
            return *this;
        }
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
     * @brief Get a view representation of the matrix's data.
     * @return CSRMatrixView for easy access to matrix elements.
     */
    [[nodiscard]] CSRMatrixView<ValueType, IndexType> view()
    {
        return CSRMatrixView(values_.span(), colIdxs_.span(), rowPtrs_.span());
    }

    /**
     * @brief Get a const view representation of the matrix's data.
     * @return Const CSRMatrixView for read-only access to matrix elements.
     */
    [[nodiscard]] const CSRMatrixView<const ValueType, const IndexType> view() const
    {
        return CSRMatrixView(values_.span(), colIdxs_.span(), rowPtrs_.span());
    }

private:

    Field<ValueType> values_;  //!< The (non-zero) values of the CSR matrix.
    Field<IndexType> colIdxs_; //!< The column indices of the CSR matrix.
    Field<IndexType> rowPtrs_; //!< The row offsets for the CSR matrix.
};

} // namespace NeoFOAM
