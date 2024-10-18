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
#include "NeoFOAM/dsl/timeIntegration/timeIntegration.hpp"


namespace NeoFOAM::dsl
{
template<typename FieldType>
void solve(
    Expression& eqn, FieldType& solution, const Dictionary& fvSchemes, const Dictionary& fvSolution
)
{
    /* @brief solve an equation
     *
     * @param solutionField - Field for which the equation is to be solved
     * @param fvSchemes - Dictionary containing spatial operator and time  integration properties
     * @param fvSolution - Dictionary containing linear solver properties
     * @tparam FieldType - type of the underlying field, e.g. VolumeField or plain Field
     */
    if (eqn.temporalOperators().size() == 0 && eqn.implicitOperators().size() == 0)
    {
        NF_ERROR_EXIT("No temporal or implicit terms to solve.");
    }
    if (eqn.temporalOperators().size() > 0)
    {
        // integrate equations in time
        TimeIntegration<FieldType> timeIntegrator(fvSchemes.subDict("ddtSchemes"));
        timeIntegrator.solve(eqn, solution);
    }
    else
    {
        NF_ERROR_EXIT("Not implemented.");
        // solve sparse matrix system
    }
}

} // namespace NeoFOAM::dsl
