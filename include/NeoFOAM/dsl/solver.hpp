// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <concepts>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/expression.hpp"
#include "NeoFOAM/timeIntegration/timeIntegration.hpp"

#if NF_WITH_GINKGO
#include "NeoFOAM/linearAlgebra/ginkgo.hpp"
#elif NF_WITH_PETSC
#include "NeoFOAM/linearAlgebra/petscSolver.hpp"
#endif


namespace NeoFOAM::dsl
{

template<typename FieldType>
NeoFOAM::la::LinearSystem<typename FieldType::ElementType, int> ginkgoMatrix(
    NeoFOAM::la::LinearSystem<typename FieldType::ElementType, localIdx>& ls, FieldType& solution
)
{
    using ValueType = typename FieldType::ElementType;
    Field<ValueType> rhs(solution.exec(), ls.rhs().data(), ls.rhs().size());

    Field<ValueType> mValues(
        solution.exec(), ls.matrix().values().data(), ls.matrix().values().size()
    );
    Field<int> mColIdxs(solution.exec(), ls.matrix().colIdxs().size());
    auto mColIdxsSpan = ls.matrix().colIdxs();
    NeoFOAM::parallelFor(
        mColIdxs, KOKKOS_LAMBDA(const size_t i) { return int(mColIdxsSpan[i]); }
    );

    Field<int> mRowPtrs(solution.exec(), ls.matrix().rowPtrs().size());
    auto mRowPtrsSpan = ls.matrix().rowPtrs();
    NeoFOAM::parallelFor(
        mRowPtrs, KOKKOS_LAMBDA(const size_t i) { return int(mRowPtrsSpan[i]); }
    );

    la::CSRMatrix<ValueType, int> matrix(mValues, mColIdxs, mRowPtrs);

    NeoFOAM::la::LinearSystem<ValueType, int> convertedLs(matrix, rhs, ls.sparsityPattern());
    return convertedLs;
}

template<typename FieldType>
NeoFOAM::la::LinearSystem<typename FieldType::ElementType, int>
ginkgoMatrix(Expression& exp, FieldType& solution)
{
    using ValueType = typename FieldType::ElementType;
    auto vol = solution.mesh().cellVolumes().span();
    auto expSource = exp.explicitOperation(solution.mesh().nCells());
    auto expSourceSpan = expSource.span();

    auto ls = exp.implicitOperation();
    Field<ValueType> rhs(solution.exec(), ls.rhs().data(), ls.rhs().size());
    auto rhsSpan = rhs.span();
    // we subtract the explicit source term from the rhs
    NeoFOAM::parallelFor(
        solution.exec(),
        {0, rhs.size()},
        KOKKOS_LAMBDA(const size_t i) { rhsSpan[i] -= expSourceSpan[i] * vol[i]; }
    );

    Field<ValueType> mValues(
        solution.exec(), ls.matrix().values().data(), ls.matrix().values().size()
    );
    Field<int> mColIdxs(solution.exec(), ls.matrix().colIdxs().size());
    auto mColIdxsSpan = ls.matrix().colIdxs();
    NeoFOAM::parallelFor(
        mColIdxs, KOKKOS_LAMBDA(const size_t i) { return int(mColIdxsSpan[i]); }
    );

    Field<int> mRowPtrs(solution.exec(), ls.matrix().rowPtrs().size());
    auto mRowPtrsSpan = ls.matrix().rowPtrs();
    NeoFOAM::parallelFor(
        mRowPtrs, KOKKOS_LAMBDA(const size_t i) { return int(mRowPtrsSpan[i]); }
    );

    auto values = ls.matrix().values();


    la::CSRMatrix<ValueType, int> matrix(mValues, mColIdxs, mRowPtrs);

    return {matrix, rhs, ls.sparsityPattern()};
}

/* @brief solve an expression
 *
 * @param exp - Expression which is to be solved/updated.
 * @param solution - Solution field, where the solution will be 'written to'.
 * @param t - the time at the start of the time step.
 * @param dt - time step for the temporal integration
 * @param fvSchemes - Dictionary containing spatial operator and time  integration properties
 * @param fvSolution - Dictionary containing linear solver properties
 */
template<typename FieldType>
void solve(
    Expression& exp,
    FieldType& solution,
    scalar t,
    scalar dt,
    [[maybe_unused]] const Dictionary& fvSchemes,
    [[maybe_unused]] const Dictionary& fvSolution
)
{
    // FIXME:
    if (exp.temporalOperators().size() == 0 && exp.spatialOperators().size() == 0)
    {
        NF_ERROR_EXIT("No temporal or implicit terms to solve.");
    }
    exp.build(fvSchemes);
    if (exp.temporalOperators().size() > 0)
    {
        // integrate equations in time
        timeIntegration::TimeIntegration<FieldType> timeIntegrator(
            fvSchemes.subDict("ddtSchemes"), fvSolution
        );
        timeIntegrator.solve(exp, solution, t, dt);
    }
    else
    {
        // solve sparse matrix system
        using ValueType = typename FieldType::ElementType;
        auto ls = ginkgoMatrix(exp, solution);

#if NF_WITH_GINKGO
        NeoFOAM::la::ginkgo::BiCGStab<ValueType> solver(solution.exec(), fvSolution);
        solver.solve(ls, solution.internalField());
#elif NF_WITH_PETSC
        NeoFOAM::la::petscSolver::petscSolver<ValueType> solver(
            solution.exec(), fvSolution, solution.db()
        );
        solver.solve(ls, solution.internalField());
#endif
    }
}

} // namespace NeoFOAM::dsl
