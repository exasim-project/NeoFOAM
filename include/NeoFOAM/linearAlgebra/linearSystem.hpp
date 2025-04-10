// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <string>

#include "NeoN/fields/field.hpp"
#include "NeoN/linearAlgebra/CSRMatrix.hpp"


namespace NeoN::la
{

/**
 * @struct LinearSystemView
 * @brief A view linear into a linear system's data.
 *
 * @tparam ValueType The value type of the linear system.
 * @tparam IndexType The index type of the linear system.
 */
template<typename ValueType, typename IndexType>
struct LinearSystemView
{
    LinearSystemView() = default;
    ~LinearSystemView() = default;

    LinearSystemView(CSRMatrixView<ValueType, IndexType> matrixView, std::span<ValueType> rhsView)
        : matrix(matrixView), rhs(rhsView) {};

    CSRMatrixView<ValueType, IndexType> matrix;
    std::span<ValueType> rhs;
};

/**
 * @class LinearSystem
 * @brief A class representing a linear system of equations.
 *
 * The LinearSystem class provides functionality to store and manipulate a linear system of
 * equations. It supports the storage of the coefficient matrix and the right-hand side vector, as
 * well as the solution vector.
 */
template<typename ValueType, typename IndexType>
class LinearSystem
{
public:

    LinearSystem(const CSRMatrix<ValueType, IndexType>& matrix, const Field<ValueType>& rhs)
        : matrix_(matrix), rhs_(rhs)
    {
        NF_ASSERT(matrix.exec() == rhs.exec(), "Executors are not the same");
        NF_ASSERT(matrix.nRows() == rhs.size(), "Matrix and RHS size mismatch");
    };

    LinearSystem(const LinearSystem& ls) : matrix_(ls.matrix_), rhs_(ls.rhs_) {};

    LinearSystem(const Executor exec) : matrix_(exec), rhs_(exec, 0) {}

    ~LinearSystem() = default;

    [[nodiscard]] CSRMatrix<ValueType, IndexType>& matrix() { return matrix_; }

    [[nodiscard]] Field<ValueType>& rhs() { return rhs_; }

    [[nodiscard]] const CSRMatrix<ValueType, IndexType>& matrix() const { return matrix_; }

    [[nodiscard]] const Field<ValueType>& rhs() const { return rhs_; }

    [[nodiscard]] LinearSystem copyToHost() const
    {
        return LinearSystem(matrix_.copyToHost(), rhs_.copyToHost());
    }

    void reset()
    {
        fill(matrix_.values(), zero<ValueType>());
        fill(rhs_, zero<ValueType>());
    }

    [[nodiscard]] LinearSystemView<ValueType, IndexType> view() && = delete;

    [[nodiscard]] LinearSystemView<ValueType, IndexType> view() const&& = delete;

    [[nodiscard]] LinearSystemView<ValueType, IndexType> view() &
    {
        return LinearSystemView<ValueType, IndexType>(matrix_.view(), rhs_.span());
    }

    [[nodiscard]] LinearSystemView<const ValueType, const IndexType> view() const&
    {
        return LinearSystemView<const ValueType, const IndexType>(matrix_.view(), rhs_.span());
    }

    const Executor& exec() const { return matrix_.exec(); }

private:

    CSRMatrix<ValueType, IndexType> matrix_;
    Field<ValueType> rhs_;
};


template<typename ValueType, typename IndexType>
Field<ValueType> spmv(LinearSystem<ValueType, IndexType>& ls, Field<ValueType>& xfield)
{
    Field<ValueType> resultField(ls.exec(), ls.rhs().size(), 0.0);
    auto [result, b, x] = spans(resultField, ls.rhs(), xfield);

    auto values = ls.matrix().values().span();
    auto colIdxs = ls.matrix().colIdxs().span();
    auto rowPtrs = ls.matrix().rowPtrs().span();

    parallelFor(
        ls.exec(),
        {0, ls.matrix().nRows()},
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
};


template<typename ValueTypeIn, typename IndexTypeIn, typename ValueTypeOut, typename IndexTypeOut>
LinearSystem<ValueTypeOut, IndexTypeOut>
convertLinearSystem(const LinearSystem<ValueTypeIn, IndexTypeIn>& ls)
{
    auto exec = ls.exec();
    Field<ValueTypeOut> convertedRhs(exec, ls.rhs().data(), ls.rhs().size());
    return {
        convert<ValueTypeIn, IndexTypeIn, ValueTypeOut, IndexTypeOut>(exec, ls.view.matrix),
        convertedRhs,
        ls.sparsityPattern()
    };
}

/*@brief helper function that creates a zero initialised linear system based on given sparsity
 * pattern
 */
template<typename ValueType, typename IndexType, typename SparsityType>
LinearSystem<ValueType, IndexType> createEmptyLinearSystem(const SparsityType& sparsity)
{
    const auto& exec = sparsity.mesh().exec();

    localIdx rows {sparsity.rows()};
    localIdx nnzs {sparsity.nnz()};

    return {
        CSRMatrix<ValueType, IndexType> {
            Field<ValueType>(exec, nnzs, zero<ValueType>()), sparsity.colIdxs(), sparsity.rowPtrs()
        },
        Field<ValueType> {exec, rows, zero<ValueType>()}
    };
}


} // namespace NeoN::la
