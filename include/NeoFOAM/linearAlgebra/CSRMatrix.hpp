// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <tuple>

#include "NeoFOAM/fields/field.hpp"


namespace NeoFOAM::la
{

/**
 * @struct CSRMatrixView
 * @brief A view struct to allow easy read/write on all executors.
 *
 * @tparam ValueType The value type of the non-zero entries.
 * @tparam IndexType The index type of the rows and columns.
 */
template<typename ValueType, typename IndexType>
struct CSRMatrixView
{
    /**
     * @brief Constructor for CSRMatrixView.
     * @param inValue Span of the non-zero values of the matrix.
     * @param inColumnIndex Span of the column indices for each non-zero value.
     * @param inRwOffset Span of the starting index in values/colIdxs for each row.
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

/**
 * @class CSRMatrix
 * @brief Sparse matrix class with compact storage by row (CSR) format.
 * @tparam ValueType The value type of the non-zero entries.
 * @tparam IndexType The index type of the rows and columns.
 */
template<typename ValueType, typename IndexType>
class CSRMatrix
{

public:

    /**
     * @brief Constructor for CSRMatrix.
     * @param value The non-zero values of the matrix.
     * @param columnIndex The column indices for each non-zero value.
     * @param rowOffset The starting index in values/colIdxs for each row.
     */
    CSRMatrix(
        const Field<ValueType>& value,
        const Field<IndexType>& columnIndex,
        const Field<IndexType>& rowOffset
    )
        : value_(value), columnIndex_(columnIndex), rowOffset_(rowOffset)
    {
        NF_ASSERT(value.exec() == columnIndex_.exec(), "Executors are not the same");
        NF_ASSERT(value.exec() == rowOffset_.exec(), "Executors are not the same");
    }

    CSRMatrix(const Executor exec) : value_(exec, 0), columnIndex_(exec, 0), rowOffset_(exec, 0) {}

    /**
     * @brief Default destructor.
     */
    ~CSRMatrix() = default;

    /**
     * @brief Get the executor associated with this matrix.
     * @return Reference to the executor.
     */
    [[nodiscard]] const Executor& exec() const { return value_.exec(); }

    /**
     * @brief Get the number of rows in the matrix.
     * @return Number of rows.
     */
    [[nodiscard]] IndexType nRows() const
    {
        return static_cast<IndexType>(rowOffset_.size())
             - static_cast<IndexType>(static_cast<bool>(rowOffset_.size()));
    }

    /**
     * @brief Get the number of non-zero values in the matrix.
     * @return Number of non-zero values.
     */
    [[nodiscard]] IndexType nNonZeros() const { return static_cast<IndexType>(value_.size()); }

    /**
     * @brief Get a span to the values array.
     * @return Span containing the matrix values.
     */
    [[nodiscard]] std::span<ValueType> values() { return value_.span(); }

    /**
     * @brief Get a span to the column indices array.
     * @return Span containing the column indices.
     */
    [[nodiscard]] std::span<IndexType> colIdxs() { return columnIndex_.span(); }

    /**
     * @brief Get a span to the row pointers array.
     * @return Span containing the row pointers.
     */
    [[nodiscard]] std::span<IndexType> rowPtrs() { return rowOffset_.span(); }

    /**
     * @brief Get a const span to the values array.
     * @return Const span containing the matrix values.
     */
    [[nodiscard]] const std::span<const ValueType> values() const { return value_.span(); }

    /**
     * @brief Get a const span to the column indices array.
     * @return Const span containing the column indices.
     */
    [[nodiscard]] const std::span<const IndexType> colIdxs() const { return columnIndex_.span(); }

    /**
     * @brief Get a const span to the row pointers array.
     * @return Const span containing the row pointers.
     */
    [[nodiscard]] const std::span<const IndexType> rowPtrs() const { return rowOffset_.span(); }

    /**
     * @brief Copy the matrix to another executor.
     * @param dstExec The destination executor.
     * @return A copy of the matrix on the destination executor.
     */
    [[nodiscard]] CSRMatrix<ValueType, IndexType> copyToExecutor(Executor dstExec) const
    {
        if (dstExec == value_.exec())
        {
            return *this;
        }
        CSRMatrix<ValueType, IndexType> other(
            value_.copyToHost(), columnIndex_.copyToHost(), rowOffset_.copyToHost()
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
        return CSRMatrixView(value_.span(), columnIndex_.span(), rowOffset_.span());
    }

    /**
     * @brief Get a const view representation of the matrix's data.
     * @return Const CSRMatrixView for read-only access to matrix elements.
     */
    [[nodiscard]] const CSRMatrixView<const ValueType, const IndexType> view() const
    {
        return CSRMatrixView(value_.span(), columnIndex_.span(), rowOffset_.span());
    }

private:

    Field<ValueType> value_;       //!< The (non-zero) values of the CSR matrix.
    Field<IndexType> columnIndex_; //!< The column indices of the CSR matrix.
    Field<IndexType> rowOffset_;   //!< The row offsets for the CSR matrix.
};

/* @brief given a csr matrix this function copies the matrix and converts to requested target types
 *
 *
 */
template<typename ValueTypeIn, typename IndexTypeIn, typename ValueTypeOut, typename IndexTypeOut>
la::CSRMatrix<ValueTypeOut, IndexTypeOut>
convert(const Executor exec, const la::CSRMatrixView<ValueTypeIn, IndexTypeIn> in)
{
    Field<IndexTypeOut> colIdxsTmp(exec, in.columnIndex.size());
    Field<IndexTypeOut> rowPtrsTmp(exec, in.rowOffset.size());
    Field<ValueTypeOut> valuesTmp(exec, in.value.data(), in.value.size());

    parallelFor(
        colIdxsTmp, KOKKOS_LAMBDA(const size_t i) { return IndexTypeOut(in.columnIndex[i]); }
    );
    parallelFor(
        rowPtrsTmp, KOKKOS_LAMBDA(const size_t i) { return IndexTypeOut(in.rowOffset[i]); }
    );
    parallelFor(
        valuesTmp, KOKKOS_LAMBDA(const size_t i) { return ValueTypeOut(in.value[i]); }
    );

    return la::CSRMatrix<ValueTypeOut, IndexTypeOut> {valuesTmp, colIdxsTmp, rowPtrsTmp};
}


} // namespace NeoFOAM
