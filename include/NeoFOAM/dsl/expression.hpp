// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <memory>
#include <vector>
#include <utility>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/dsl/spatialOperator.hpp"
#include "NeoFOAM/dsl/temporalOperator.hpp"
#include "NeoFOAM/core/error.hpp"

namespace la = NeoFOAM::la;

namespace NeoFOAM::dsl
{


class Expression
{
public:

    Expression(const Executor& exec) : exec_(exec), temporalOperators_(), spatialOperators_() {}

    Expression(const Expression& exp)
        : exec_(exp.exec_), temporalOperators_(exp.temporalOperators_),
          spatialOperators_(exp.spatialOperators_)
    {}

    void build(const NeoFOAM::Dictionary& input)
    {
        for (auto& op : temporalOperators_)
        {
            op.build(input);
        }
        for (auto& op : spatialOperators_)
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
        for (auto& oper : spatialOperators_)
        {
            if (oper.getType() == SpatialOperator::Type::Explicit)
            {
                oper.explicitOperation(source);
            }
        }
        // for (auto& oper : temporalOperators_)
        // {
        //     if (oper.getType() == SpatialOperator::Type::Explicit)
        //     {
        //         oper.explicitOperation(source);
        //     }
        // }
        return source;
    }

    /* @brief perform all implicit operation and accumulate the result */
    la::LinearSystem<scalar, localIdx> implicitOperation()
    {
        // TODO: implement
        // if (implicitOperators_.empty())
        // {
        //     NF_ERROR_EXIT("No implicit operators in the expression");
        // }
        auto ls = spatialOperators_[0].createEmptyLinearSystem();
        for (auto& oper : spatialOperators_)
        {
            if (oper.getType() == SpatialOperator::Type::Implicit)
            {
                oper.implicitOperation(ls);
            }
        }
        return ls;
    }

    void addOperator(const SpatialOperator& oper) { spatialOperators_.push_back(oper); }

    void addOperator(const TemporalOperator& oper) { temporalOperators_.push_back(oper); }

    void addExpression(const Expression& equation)
    {
        for (auto& oper : equation.temporalOperators_)
        {
            temporalOperators_.push_back(oper);
        }
        for (auto& oper : equation.spatialOperators_)
        {
            spatialOperators_.push_back(oper);
        }
    }


    /* @brief getter for the total number of terms in the equation */
    size_t size() const { return temporalOperators_.size() + spatialOperators_.size(); }

    // getters
    const std::vector<TemporalOperator>& temporalOperators() const { return temporalOperators_; }

    const std::vector<SpatialOperator>& spatialOperators() const { return spatialOperators_; }

    std::vector<TemporalOperator>& temporalOperators() { return temporalOperators_; }

    std::vector<SpatialOperator>& spatialOperators() { return spatialOperators_; }

    const Executor& exec() const { return exec_; }

private:

    const Executor exec_;

    std::vector<TemporalOperator> temporalOperators_;

    std::vector<SpatialOperator> spatialOperators_;
};

[[nodiscard]] inline Expression operator+(Expression lhs, const Expression& rhs)
{
    lhs.addExpression(rhs);
    return lhs;
}

[[nodiscard]] inline Expression operator+(Expression lhs, const SpatialOperator& rhs)
{
    lhs.addOperator(rhs);
    return lhs;
}

template<typename leftOperator, typename rightOperator>
[[nodiscard]] inline Expression operator+(leftOperator lhs, rightOperator rhs)
{
    Expression expr(lhs.exec());
    expr.addOperator(lhs);
    expr.addOperator(rhs);
    return expr;
}

[[nodiscard]] inline Expression operator+(const SpatialOperator& lhs, const SpatialOperator& rhs)
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
    for (const auto& oper : es.spatialOperators())
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

[[nodiscard]] inline Expression operator-(Expression lhs, const SpatialOperator& rhs)
{
    lhs.addOperator(-1.0 * rhs);
    return lhs;
}

[[nodiscard]] inline Expression operator-(const SpatialOperator& lhs, const SpatialOperator& rhs)
{
    Expression expr(lhs.exec());
    expr.addOperator(lhs);
    expr.addOperator(Coeff(-1) * rhs);
    return expr;
}


} // namespace NeoFOAM::dsl
