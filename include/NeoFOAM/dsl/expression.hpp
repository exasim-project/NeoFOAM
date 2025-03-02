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


template<typename ValueType>
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
    Field<ValueType> explicitOperation(size_t nCells)
    {
        Field<ValueType> source(exec_, nCells, 0.0);
        return explicitOperation(source);
    }

    /* @brief perform all explicit operation and accumulate the result */
    Field<ValueType> explicitOperation(Field<ValueType>& source)
    {
        for (auto& op : spatialOperators_)
        {
            if (op.getType() == Operator::Type::Explicit)
            {
                op.explicitOperation(source);
            }
        }
        return source;
    }

    Field<ValueType> explicitOperation(Field<ValueType>& source, scalar t, scalar dt)
    {
        for (auto& op : temporalOperators_)
        {
            if (op.getType() == Operator::Type::Explicit)
            {
                op.explicitOperation(source, t, dt);
            }
        }
        return source;
    }

    /* @brief perform all implicit operation and accumulate the result */
    la::LinearSystem<ValueType, localIdx> implicitOperation()
    {
        auto ls = spatialOperators_[0].createEmptyLinearSystem();
        for (auto& op : spatialOperators_)
        {
            if (op.getType() == Operator::Type::Implicit)
            {
                op.implicitOperation(ls);
            }
        }
        return ls;
    }

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls, scalar t, scalar dt)
    {
        for (auto& op : temporalOperators_)
        {
            if (op.getType() == Operator::Type::Implicit)
            {
                op.implicitOperation(ls, t, dt);
            }
        }
    }


    void addOperator(const SpatialOperator<ValueType>& oper) { spatialOperators_.push_back(oper); }

    void addOperator(const TemporalOperator<ValueType>& oper)
    {
        temporalOperators_.push_back(oper);
    }

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
    const std::vector<TemporalOperator<ValueType>>& temporalOperators() const
    {
        return temporalOperators_;
    }

    const std::vector<SpatialOperator<ValueType>>& spatialOperators() const
    {
        return spatialOperators_;
    }

    std::vector<TemporalOperator<ValueType>>& temporalOperators() { return temporalOperators_; }

    std::vector<SpatialOperator<ValueType>>& spatialOperators() { return spatialOperators_; }

    const Executor& exec() const { return exec_; }

private:

    const Executor exec_;

    std::vector<TemporalOperator<ValueType>> temporalOperators_;

    std::vector<SpatialOperator<ValueType>> spatialOperators_;
};

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator+(Expression<ValueType> lhs, const Expression<ValueType>& rhs)
{
    lhs.addExpression(rhs);
    return lhs;
}

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator+(Expression<ValueType> lhs, const SpatialOperator<ValueType>& rhs)
{
    lhs.addOperator(rhs);
    return lhs;
}

template<typename leftOperator, typename rightOperator>
[[nodiscard]] inline Expression<typename leftOperator::FieldValueType>
operator+(leftOperator lhs, rightOperator rhs)
{
    using ValueType = typename leftOperator::FieldValueType;
    Expression<ValueType> expr(lhs.exec());
    expr.addOperator(lhs);
    expr.addOperator(rhs);
    return expr;
}

// [[nodiscard]] inline Expression operator+(const SpatialOperator& lhs, const SpatialOperator& rhs)
// {
//     Expression expr(lhs.exec());
//     expr.addOperator(lhs);
//     expr.addOperator(rhs);
//     return expr;
// }

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType> operator*(scalar scale, const Expression<ValueType>& es)
{
    Expression<ValueType> expr(es.exec());
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


template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator-(Expression<ValueType> lhs, const Expression<ValueType>& rhs)
{
    lhs.addExpression(-1.0 * rhs);
    return lhs;
}

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator-(Expression<ValueType> lhs, const SpatialOperator<ValueType>& rhs)
{
    lhs.addOperator(-1.0 * rhs);
    return lhs;
}

template<typename leftOperator, typename rightOperator>
[[nodiscard]] inline Expression<typename leftOperator::FieldValueType>
operator-(leftOperator lhs, rightOperator rhs)
{
    using ValueType = typename leftOperator::FieldValueType;
    Expression<ValueType> expr(lhs.exec());
    expr.addOperator(lhs);
    expr.addOperator(Coeff(-1) * rhs);
    return expr;
}

// [[nodiscard]] inline Expression operator-(const SpatialOperator& lhs, const SpatialOperator& rhs)
// {
//     Expression expr(lhs.exec());
//     expr.addOperator(lhs);
//     expr.addOperator(Coeff(-1) * rhs);
//     return expr;
// }


} // namespace NeoFOAM::dsl
