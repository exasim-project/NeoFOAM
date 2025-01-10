// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <memory>
#include <vector>
#include <utility>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/dsl/operator.hpp"
#include "NeoFOAM/core/error.hpp"

namespace NeoFOAM::dsl
{


class Expression
{
public:

    Expression(const Executor& exec)
        : exec_(exec), temporalOperators_(), implicitOperators_(), explicitOperators_()
    {}

    Expression(const Expression& exp)
        : exec_(exp.exec_), temporalOperators_(exp.temporalOperators_),
          implicitOperators_(exp.implicitOperators_), explicitOperators_(exp.explicitOperators_)
    {}

    void build(const NeoFOAM::Dictionary& input)
    {
        for (auto& op : temporalOperators_)
        {
            op.build(input);
        }
        for (auto& op : implicitOperators_)
        {
            op.build(input);
        }
        for (auto& op : explicitOperators_)
        {
            op.build(input);
        }
    }

    /* @brief perform all explicit operation and accumulate the result */
    Field<scalar> explicitOperation(size_t nCells)
    {
        Field<scalar> source(exec_, nCells, 0.0);
        return explicitOperation(source);
    }

    /* @brief perform all explicit operation and accumulate the result */
    Field<scalar> explicitOperation(Field<scalar>& source)
    {
        for (auto& oper : explicitOperators_)
        {
            oper.explicitOperation(source);
        }
        return source;
    }

    void addOperator(const Operator& oper)
    {
        switch (oper.getType())
        {
        case Operator::Type::Temporal:
            temporalOperators_.push_back(oper);
            break;
        case Operator::Type::Implicit:
            implicitOperators_.push_back(oper);
            break;
        case Operator::Type::Explicit:
            explicitOperators_.push_back(oper);
            break;
        }
    }

    void addExpression(const Expression& equation)
    {
        for (auto& oper : equation.temporalOperators_)
        {
            temporalOperators_.push_back(oper);
        }
        for (auto& oper : equation.implicitOperators_)
        {
            implicitOperators_.push_back(oper);
        }
        for (auto& oper : equation.explicitOperators_)
        {
            explicitOperators_.push_back(oper);
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

private:

    const Executor exec_;

    std::vector<Operator> temporalOperators_;

    std::vector<Operator> implicitOperators_;

    std::vector<Operator> explicitOperators_;
};

[[nodiscard]] inline Expression operator+(Expression lhs, const Expression& rhs)
{
    lhs.addExpression(rhs);
    return lhs;
}

[[nodiscard]] inline Expression operator+(Expression lhs, const Operator& rhs)
{
    lhs.addOperator(rhs);
    return lhs;
}

[[nodiscard]] inline Expression operator+(const Operator& lhs, const Operator& rhs)
{
    Expression expr(lhs.exec());
    expr.addOperator(lhs);
    expr.addOperator(rhs);
    return expr;
}

[[nodiscard]] inline Expression operator*(scalar scale, const Expression& es)
{
    Expression expr(es.exec());
    for (const auto& oper : es.temporalOperators())
    {
        expr.addOperator(scale * oper);
    }
    for (const auto& oper : es.implicitOperators())
    {
        expr.addOperator(scale * oper);
    }
    for (const auto& oper : es.explicitOperators())
    {
        expr.addOperator(scale * oper);
    }
    return expr;
}

[[nodiscard]] inline Expression operator-(Expression lhs, const Expression& rhs)
{
    lhs.addExpression(-1.0 * rhs);
    return lhs;
}

[[nodiscard]] inline Expression operator-(Expression lhs, const Operator& rhs)
{
    lhs.addOperator(-1.0 * rhs);
    return lhs;
}

[[nodiscard]] inline Expression operator-(const Operator& lhs, const Operator& rhs)
{
    Expression expr(lhs.exec());
    expr.addOperator(lhs);
    expr.addOperator(Coeff(-1) * rhs);
    return expr;
}


} // namespace NeoFOAM::dsl
