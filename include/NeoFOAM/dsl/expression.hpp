// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <utility>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/dsl/operator.hpp"
#include "NeoFOAM/core/error.hpp"

#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoFOAM::dsl
{


class Expression
{
public:

    Expression(const Executor& exec, std::size_t nCells)
        : exec_(exec), nCells_(nCells), temporalOperators_(), implicitOperators_(),
          explicitOperators_()
    {}

    /* @brief perform all explicit operation and accumulate the result */
    Field<scalar> explicitOperation() const
    {
        Field<scalar> source(exec_, nCells_, 0.0);
        return explicitOperation(source);
    }

    /* @brief perform all explicit operation and accumulate the result */
    Field<scalar> explicitOperation(Field<scalar>& source) const
    {
        for (auto& Operator : explicitOperators_)
        {
            Operator.explicitOperation(source);
        }
        return source;
    }

    void addOperator(const Operator& Operator)
    {
        switch (Operator.getType())
        {
        case Operator::Type::Temporal:
            temporalOperators_.push_back(Operator);
            break;
        case Operator::Type::Implicit:
            implicitOperators_.push_back(Operator);
            break;
        case Operator::Type::Explicit:
            explicitOperators_.push_back(Operator);
            break;
        }
    }

    void addExpression(const Expression& equation)
    {
        for (auto& Operator : equation.temporalOperators_)
        {
            temporalOperators_.push_back(Operator);
        }
        for (auto& Operator : equation.implicitOperators_)
        {
            implicitOperators_.push_back(Operator);
        }
        for (auto& Operator : equation.explicitOperators_)
        {
            explicitOperators_.push_back(Operator);
        }
    }


    /* @brief getter for the total number of terms in the equation */
    size_t size() const
    {
        return temporalOperators_.size() + implicitOperators_.size() + explicitOperators_.size();
    }

    // getters
    const std::vector<Operator>& temporalOperators() const { return temporalOperators_; }

    const std::vector<Operator>& implicitOperators() const { return implicitOperators_; }

    const std::vector<Operator>& explicitOperators() const { return explicitOperators_; }

    std::vector<Operator>& temporalOperators() { return temporalOperators_; }

    std::vector<Operator>& implicitOperators() { return implicitOperators_; }

    std::vector<Operator>& explicitOperators() { return explicitOperators_; }

    const Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

    scalar getDt() const { return dt_; }

    void setDt(scalar dt) { dt_ = dt; }

private:

    scalar dt_ = 0;

    const Executor exec_;

    const std::size_t nCells_;

    std::vector<Operator> temporalOperators_;

    std::vector<Operator> implicitOperators_;

    std::vector<Operator> explicitOperators_;
};

Expression operator+(Expression lhs, const Expression& rhs)
{
    lhs.addExpression(rhs);
    return lhs;
}

Expression operator+(Expression lhs, const Operator& rhs)
{
    lhs.addOperator(rhs);
    return lhs;
}

Expression operator+(const Operator& lhs, const Operator& rhs)
{
    Expression equation(lhs.exec(), lhs.getSize());
    equation.addOperator(lhs);
    equation.addOperator(rhs);
    return equation;
}

Expression operator*(scalar scale, const Expression& es)
{
    Expression results(es.exec(), es.nCells());
    for (const auto& Operator : es.temporalOperators())
    {
        results.addOperator(scale * Operator);
    }
    for (const auto& Operator : es.implicitOperators())
    {
        results.addOperator(scale * Operator);
    }
    for (const auto& Operator : es.explicitOperators())
    {
        results.addOperator(scale * Operator);
    }
    return results;
}

Expression operator-(Expression lhs, const Expression& rhs)
{
    lhs.addExpression(-1.0 * rhs);
    return lhs;
}

Expression operator-(Expression lhs, const Operator& rhs)
{
    lhs.addOperator(-1.0 * rhs);
    return lhs;
}

Expression operator-(const Operator& lhs, const Operator& rhs)
{
    Expression equation(lhs.exec(), lhs.getSize());
    equation.addOperator(lhs);
    equation.addOperator(Coeff(-1) * rhs);
    return equation;
}


} // namespace NeoFOAM::dsl
