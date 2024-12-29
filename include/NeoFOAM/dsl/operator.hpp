// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <memory>
#include <concepts>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/coeff.hpp"

namespace NeoFOAM::dsl
{

template<typename T>
concept HasTemporalOperator = requires(T t) {
    {
        t.temporalOperation(std::declval<Field<scalar>&>(), std::declval<scalar>())
    } -> std::same_as<void>; // Adjust return type and arguments as needed
};

template<typename T>
concept HasExplicitOperator = requires(T t) {
    {
        t.explicitOperation(std::declval<Field<scalar>&>())
    } -> std::same_as<void>; // Adjust return type and arguments as needed
};

/* @class Operator
 * @brief A class to represent an operator in NeoFOAMs dsl
 *
 * The design here is based on the type erasure design pattern
 * see https://www.youtube.com/watch?v=4eeESJQk-mw
 *
 * Motivation for using type erasure is that concrete implementation
 * of Operators e.g Divergence, Laplacian, etc can be stored in a vector of
 * Operators
 *
 * @ingroup dsl
 */
class Operator
{
public:

    enum class Type
    {
        Temporal,
        Implicit,
        Explicit
    };

    template<typename T>
    Operator(T cls) : model_(std::make_unique<OperatorModel<T>>(std::move(cls)))
    {}

    Operator(const Operator& eqnOperator);

    Operator(Operator&& eqnOperator);

    void explicitOperation(Field<scalar>& source);

    void temporalOperation(Field<scalar>& field);

    /* returns the fundamental type of an operator, ie explicit, implicit, temporal */
    Operator::Type getType() const;

    Coeff& getCoefficient();

    Coeff getCoefficient() const;

    /* @brief Given an input this function reads required coeffs */
    void build(const Input& input);

    /* @brief Get the executor */
    const Executor& exec() const;


private:

    /* @brief Base class defining the concept of a term. This effectively
     * defines what functions need to be implemented by a concrete Operator implementation
     * */
    struct OperatorConcept
    {
        virtual ~OperatorConcept() = default;

        virtual void explicitOperation(Field<scalar>& source) = 0;

        virtual void temporalOperation(Field<scalar>& field) = 0;

        /* @brief Given an input this function reads required coeffs */
        virtual void build(const Input& input) = 0;

        /* returns the name of the operator */
        virtual std::string getName() const = 0;

        /* returns the fundamental type of an operator, ie explicit, implicit, temporal */
        virtual Operator::Type getType() const = 0;

        /* @brief get the associated coefficient for this term */
        virtual Coeff& getCoefficient() = 0;

        /* @brief get the associated coefficient for this term */
        virtual Coeff getCoefficient() const = 0;

        /* @brief Get the executor */
        virtual const Executor& exec() const = 0;

        // The Prototype Design Pattern
        virtual std::unique_ptr<OperatorConcept> clone() const = 0;
    };

    // Templated derived class to implement the type-specific behavior
    template<typename ConcreteOperatorType>
    struct OperatorModel : OperatorConcept
    {
        /* @brief build with concrete operator */
        OperatorModel(ConcreteOperatorType concreteOp) : concreteOp_(std::move(concreteOp)) {}

        /* returns the name of the operator */
        std::string getName() const override { return concreteOp_.getName(); }

        virtual void explicitOperation(Field<scalar>& source) override
        {
            if constexpr (HasExplicitOperator<ConcreteOperatorType>)
            {
                concreteOp_.explicitOperation(source);
            }
        }

        virtual void temporalOperation(Field<scalar>& field) override
        {
            if constexpr (HasTemporalOperator<ConcreteOperatorType>)
            {
                concreteOp_.temporalOperation(field);
            }
        }

        /* @brief Given an input this function reads required coeffs */
        virtual void build(const Input& input) override { concreteOp_.build(input); }

        /* returns the fundamental type of an operator, ie explicit, implicit, temporal */
        Operator::Type getType() const override { return concreteOp_.getType(); }

        /* @brief Get the executor */
        const Executor& exec() const override { return concreteOp_.exec(); }

        /* @brief get the associated coefficient for this term */
        virtual Coeff& getCoefficient() override { return concreteOp_.getCoefficient(); }

        /* @brief get the associated coefficient for this term */
        virtual Coeff getCoefficient() const override { return concreteOp_.getCoefficient(); }

        // The Prototype Design Pattern
        std::unique_ptr<OperatorConcept> clone() const override
        {
            return std::make_unique<OperatorModel>(*this);
        }

        ConcreteOperatorType concreteOp_;
    };

    std::unique_ptr<OperatorConcept> model_;
};


Operator operator*(scalar scalarCoeff, Operator rhs);

Operator operator*(const Field<scalar>& coeffField, Operator rhs);

Operator operator*(const Coeff& coeff, Operator rhs);

template<typename CoeffFunction>
    requires std::invocable<CoeffFunction&, size_t>
Operator operator*([[maybe_unused]] CoeffFunction coeffFunc, const Operator& lhs)
{
    // TODO implement
    NF_ERROR_EXIT("Not implemented");
    Operator result = lhs;
    // if (!result.getCoefficient().useSpan)
    // {
    //     result.setField(std::make_shared<Field<scalar>>(result.exec(), result.nCells(), 1.0));
    // }
    // map(result.exec(), result.getCoefficient().values, scaleFunc);
    return result;
}

/* @class OperatorMixin
 * @brief A mixin class to simplify implementations of concrete operators
 * in NeoFOAMs dsl
 *
 * @ingroup dsl
 */
template<typename FieldType>
class OperatorMixin
{

public:

    OperatorMixin(const Executor exec, FieldType& field, Operator::Type type)
        : exec_(exec), coeffs_(), field_(field), type_(type) {};

    Operator::Type getType() const { return type_; }

    virtual ~OperatorMixin() = default;

    virtual const Executor& exec() const final { return exec_; }

    Coeff& getCoefficient() { return coeffs_; }

    const Coeff& getCoefficient() const { return coeffs_; }

    FieldType& getField() { return field_; }

    const FieldType& getField() const { return field_; }

    /* @brief Given an input this function reads required coeffs */
    void build([[maybe_unused]] const Input& input) {}

protected:

    const Executor exec_; //!< Executor associated with the field. (CPU, GPU, openMP, etc.)

    Coeff coeffs_;

    FieldType& field_;

    Operator::Type type_;
};
} // namespace NeoFOAM::dsl
