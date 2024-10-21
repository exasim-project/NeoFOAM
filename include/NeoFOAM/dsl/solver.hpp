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


namespace NeoFOAM::dsl
{
    /* @brief solve an expresion
     *
     * @param exp - Expression which is to be solved/updated.
     * @param solution - Solution field, where the solution will be 'written to'.
     * @param fvSchemes - Dictionary containing spatial operator and time  integration properties
     * @param fvSolution - Dictionary containing linear solver properties
     */
template<typename FieldType>
void solve(
    Expression& exp, FieldType& solution, const Dictionary& fvSchemes, const Dictionary& fvSolution
)
{
    if (exp.temporalOperators().size() == 0 && exp.implicitOperators().size() == 0)
    {
        NF_ERROR_EXIT("No temporal or implicit terms to solve.");
    }
    if (exp.temporalOperators().size() > 0)
    {
        // integrate equations in time
        TimeIntegration<FieldType> timeIntegrator(fvSchemes.subDict("ddtSchemes"));
        timeIntegrator.solve(exp, solution);
    }
    else
    {
        NF_ERROR_EXIT("Not implemented.");
        // solve sparse matrix system
    }
}

} // namespace NeoFOAM::dsl
