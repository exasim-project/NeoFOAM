// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <string>
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/linearAlgebra/CSRMatrix.hpp"


namespace NeoFOAM::la
{

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

    LinearSystem(
        const CSRMatrix<ValueType, IndexType>& matrix,
        const Field<ValueType>& rhs,
        const std::string& sparsityPattern
    )
        : matrix_(matrix), rhs_(rhs), sparsityPattern_(sparsityPattern)
    {
        NF_ASSERT(matrix.exec() == rhs.exec(), "Executors are not the same");
        NF_ASSERT(matrix.nRows() == rhs.size(), "Matrix and RHS size mismatch");
    };

    LinearSystem(const LinearSystem& ls)
        : matrix_(ls.matrix_), rhs_(ls.rhs_), sparsityPattern_(ls.sparsityPattern_) {};

    LinearSystem(const Executor exec) : matrix_(exec), rhs_(exec, 0), sparsityPattern_() {}

    ~LinearSystem() = default;

    [[nodiscard]] CSRMatrix<ValueType, IndexType>& matrix() { return matrix_; }
    [[nodiscard]] Field<ValueType>& rhs() { return rhs_; }

    [[nodiscard]] const CSRMatrix<ValueType, IndexType>& matrix() const { return matrix_; }
    [[nodiscard]] const Field<ValueType>& rhs() const { return rhs_; }

    [[nodiscard]] std::string sparsityPattern() const { return sparsityPattern_; }

    [[nodiscard]] LinearSystem copyToHost() const
    {
        return LinearSystem(matrix_.copyToHost(), rhs_.copyToHost(), sparsityPattern_);
    }

    const Executor& exec() const { return matrix_.exec(); }

private:

    CSRMatrix<ValueType, IndexType> matrix_;
    Field<ValueType> rhs_;
    std::string sparsityPattern_;
};


template<typename ValueType, typename IndexType>
Field<ValueType> SpMV(LinearSystem<ValueType, IndexType>& ls, Field<ValueType>& xfield)
{
    Field<ValueType> resultField(ls.exec(), ls.rhs().size(), 0.0);
    auto [result, b, x] = spans(resultField, ls.rhs(), xfield);

    std::span<ValueType> values = ls.matrix().values();
    std::span<IndexType> colIdxs = ls.matrix().colIdxs();
    std::span<IndexType> rowPtrs = ls.matrix().rowPtrs();

    NeoFOAM::parallelFor(
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

} // namespace NeoFOAM::la
