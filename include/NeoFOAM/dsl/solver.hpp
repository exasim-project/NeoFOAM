// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
#pragma once

#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <concepts>

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/dsl/expression.hpp"
#include "NeoN/timeIntegration/timeIntegration.hpp"

#if NF_WITH_GINKGO
#include "NeoN/linearAlgebra/ginkgo.hpp"
#endif
#include "NeoN/linearAlgebra/linearSystem.hpp"

// FIXME
#include "NeoN/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"


namespace NeoN::dsl
{

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
    Expression<typename FieldType::ElementType>& exp,
    FieldType& solution,
    scalar t,
    scalar dt,
    [[maybe_unused]] const Dictionary& fvSchemes,
    [[maybe_unused]] const Dictionary& fvSolution
)
{
    // TODO:
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

        auto sparsity = NeoN::finiteVolume::cellCentred::SparsityPattern(solution.mesh());
        auto ls = la::createEmptyLinearSystem<
            ValueType,
            localIdx,
            NeoN::finiteVolume::cellCentred::SparsityPattern>(sparsity);

        exp.implicitOperation(ls);
        auto expTmp = exp.explicitOperation(solution.mesh().nCells());

        auto [vol, expSource, rhs] = spans(solution.mesh().cellVolumes(), expTmp, ls.rhs());

        // subtract the explicit source term from the rhs
        parallelFor(
            solution.exec(),
            {0, rhs.size()},
            KOKKOS_LAMBDA(const size_t i) { rhs[i] -= expSource[i] * vol[i]; }
        );

#if NF_WITH_GINKGO
        auto solver = la::ginkgo::Solver<ValueType>(solution.exec(), fvSolution);
        solver.solve(ls, solution.internalField());
#else
        NF_ERROR_EXIT("No linear solver is available, build with -DNeoN_WITH_GINKGO=ON");
#endif
    }
}

} // namespace dsl
