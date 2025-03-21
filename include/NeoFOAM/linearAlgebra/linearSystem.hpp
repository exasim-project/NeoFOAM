// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <string>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/linearAlgebra/CSRMatrix.hpp"


namespace NeoFOAM::la
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

    LinearSystemView(
        CSRMatrixView<ValueType, IndexType> inA, std::span<ValueType> inB
    )
        : A(inA), b(inB) {};

    CSRMatrixView<ValueType, IndexType> A;
    std::span<ValueType> b;
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

    [[nodiscard]] LinearSystemView<ValueType, IndexType> view() && = delete;

    [[nodiscard]] LinearSystemView<ValueType, IndexType> view() const&& = delete;

    [[nodiscard]] LinearSystemView<ValueType, IndexType> view() &
    {
        return LinearSystemView<ValueType, IndexType>(
            matrix_.view(), rhs_.span() //, sparsityPattern_
        );
    }

    [[nodiscard]] LinearSystemView<const ValueType, const IndexType> view() const&
    {
        return LinearSystemView<const ValueType, const IndexType>(
            matrix_.view(), rhs_.span() //, sparsityPattern_
        );
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
    return {convert<ValueTypeIn, IndexTypeIn, ValueTypeOut, IndexTypeOut> (exec, ls.view.A), convertedRhs, ls.sparsityPattern()};
}



/* @brief given an expression and a solution field this function creates a linear system
 */
/*template<typename FieldType>*/
/*LinearSystem<typename FieldType::ElementType, int>*/
/*convertToLinearSystem(*/
/*		Expression<typename FieldType::ElementType>& exp,*/
/*		FieldType& solution)*/
/*{*/
/*    using ValueType = typename FieldType::ElementType;*/
/**/
/*    auto expTmp = exp.explicitOperation(solution.mesh().nCells());*/
/*    Field<ValueType> rhsOut(solution.exec(), ls.rhs().data(), ls.rhs().size());*/
/**/
/*    auto [vol, expSource, rhs] = spans(solution.mesh().cellVolumes(), expTmp, rhsOut);*/
/**/
/*    // subtract the explicit source term from the rhs*/
/*    parallelFor(*/
/*        solution.exec(),*/
/*        {0, rhs.size()},*/
/*        KOKKOS_LAMBDA(const size_t i) { rhs[i] -= expSource[i] * vol[i]; }*/
/*    );*/
/**/
/*    return {convert<ValueType,localIdx,ValueType,int>(solution.exec(), ls.view().A), rhsOut, ls.sparsityPattern()};*/
/*}*/

// FIXME consolidate with linearSystem convert
// FIXME is that needed? this seems to just make sure that row and col are in ints
/*template<typename FieldType>*/
/*LinearSystem<typename FieldType::ElementType, int>*/
/*ginkgoMatrix(LinearSystem<typename FieldType::ElementType, localIdx>& ls, FieldType& solution)*/
/*{*/
/*    using ValueType = typename FieldType::ElementType;*/
/*    Field<ValueType> rhs(solution.exec(), ls.rhs().data(), ls.rhs().size());*/
/**/
/*    LinearSystem<ValueType, int> convertedLs(convert<ValueType,localIdx,ValueType,int>(*/
/*			    solution.exec(),*/
/*			    ls.view().A*/
/*			    ), rhs, ls.sparsityPattern());*/
/*    return convertedLs;*/
/*}*/


} // namespace NeoFOAM::la
