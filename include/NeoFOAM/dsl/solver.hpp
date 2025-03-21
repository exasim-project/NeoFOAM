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
#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/dsl/expression.hpp"
#include "NeoFOAM/timeIntegration/timeIntegration.hpp"

#if NF_WITH_GINKGO
#include "NeoFOAM/linearAlgebra/ginkgo.hpp"
#endif
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"


namespace NeoFOAM::dsl
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

        auto lsTmp = exp.implicitOperation();
        auto expTmp = exp.explicitOperation(solution.mesh().nCells());

        auto [vol, expSource, rhs] = spans(solution.mesh().cellVolumes(), expTmp, lsTmp.rhs());

        // subtract the explicit source term from the rhs
        parallelFor(
            solution.exec(),
            {0, rhs.size()},
            KOKKOS_LAMBDA(const size_t i) { rhs[i] -= expSource[i] * vol[i]; }
        );

        auto solver = la::ginkgo::Solver<ValueType>(solution.exec(), fvSolution);
        solver.solve(lsTmp, solution.internalField());
    }
}

} // namespace dsl
