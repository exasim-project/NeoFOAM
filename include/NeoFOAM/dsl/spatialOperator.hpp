// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors
#pragma once

#include <memory>
#include <concepts>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/coeff.hpp"
#include "NeoFOAM/dsl/operator.hpp"

namespace la = NeoFOAM::la;

namespace NeoFOAM::dsl
{

template<typename T>
concept HasExplicitOperator = requires(T t) {
    // using ValueType = typename T::FieldValueType;
    {
        t.explicitOperation(std::declval<Field<typename T::FieldValueType>&>())
    } -> std::same_as<void>; // Adjust return type and arguments as needed
};

template<typename T>
concept HasImplicitOperator = requires(T t) {
    // using ValueType = typename T::FieldValueType;
    {
        t.implicitOperation(std::declval<la::LinearSystem<typename T::FieldValueType, localIdx>&>())
    } -> std::same_as<void>; // Adjust return type and arguments as needed
};

template<typename T>
concept IsSpatialOperator = HasExplicitOperator<T> || HasImplicitOperator<T>;

/* @class SpatialOperator
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
template<typename ValueType>
class SpatialOperator
{
public:

    using FieldValueType = ValueType;

    template<IsSpatialOperator T>
    SpatialOperator(T cls) : model_(std::make_unique<OperatorModel<T>>(std::move(cls)))
    {}

    SpatialOperator(const SpatialOperator& eqnOperator) : model_(eqnOperator.model_->clone()) {}

    SpatialOperator(SpatialOperator&& eqnOperator) : model_(std::move(eqnOperator.model_)) {}

    SpatialOperator& operator=(const SpatialOperator& eqnOperator)
    {
        model_ = eqnOperator.model_->clone();
        return *this;
    }

    void explicitOperation(Field<ValueType>& source) { model_->explicitOperation(source); }

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls)
    {
        model_->implicitOperation(ls);
    }

    la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const
    {
        return model_->createEmptyLinearSystem();
    }

    /* returns the fundamental type of an operator, ie explicit, implicit */
    Operator::Type getType() const { return model_->getType(); }

    std::string getName() const { return model_->getName(); }


    Coeff& getCoefficient() { return model_->getCoefficient(); }

    Coeff getCoefficient() const { return model_->getCoefficient(); }

    /* @brief Given an input this function reads required coeffs */
    void build(const Input& input) { model_->build(input); }

    /* @brief Get the executor */
    const Executor& exec() const { return model_->exec(); }


private:

    /* @brief Base class defining the concept of a term. This effectively
     * defines what functions need to be implemented by a concrete Operator implementation
     * */
    struct OperatorConcept
    {
        virtual ~OperatorConcept() = default;

        virtual void explicitOperation(Field<ValueType>& source) = 0;

        virtual void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) = 0;

        virtual la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const = 0;

        /* @brief Given an input this function reads required coeffs */
        virtual void build(const Input& input) = 0;

        /* returns the name of the operator */
        virtual std::string getName() const = 0;

        /* returns the fundamental type of an operator, ie explicit, implicit */
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

        virtual void explicitOperation(Field<ValueType>& source) override
        {
            if constexpr (HasExplicitOperator<ConcreteOperatorType>)
            {
                concreteOp_.explicitOperation(source);
            }
        }

        virtual void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) override
        {
            if constexpr (HasImplicitOperator<ConcreteOperatorType>)
            {
                concreteOp_.implicitOperation(ls);
            }
        }

        virtual la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const override
        {
            if constexpr (HasImplicitOperator<ConcreteOperatorType>)
            {
                return concreteOp_.createEmptyLinearSystem();
            }
            throw std::runtime_error("Implicit operation not implemented");
            // only need to avoid compiler warning about missing return statement
            // this code path should never be reached as we call implicitOperation on an explicit
            // operator
            Field<ValueType> values(exec(), 1, zero<ValueType>());
            Field<localIdx> colIdx(exec(), 1, 0);
            Field<localIdx> rowPtrs(exec(), 2, 0);
            la::CSRMatrix<ValueType, localIdx> csrMatrix(values, colIdx, rowPtrs);

            Field<ValueType> rhs(exec(), 1, zero<ValueType>());
            return la::LinearSystem<ValueType, localIdx>(csrMatrix, rhs, "custom");
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


template<typename ValueType>
SpatialOperator<ValueType> operator*(scalar scalarCoeff, SpatialOperator<ValueType> rhs)
{
    SpatialOperator<ValueType> result = rhs;
    result.getCoefficient() *= scalarCoeff;
    return result;
}

template<typename ValueType>
SpatialOperator<ValueType>
operator*(const Field<scalar>& coeffField, SpatialOperator<ValueType> rhs)
{
    SpatialOperator<ValueType> result = rhs;
    result.getCoefficient() *= Coeff(coeffField);
    return result;
}

template<typename ValueType>
SpatialOperator<ValueType> operator*(const Coeff& coeff, SpatialOperator<ValueType> rhs)
{
    SpatialOperator<ValueType> result = rhs;
    result.getCoefficient() *= coeff;
    return result;
}

// template<typename CoeffFunction>
//     requires std::invocable<CoeffFunction&, size_t>
// SpatialOperator operator*([[maybe_unused]] CoeffFunction coeffFunc, const SpatialOperator& lhs)
// {
//     // TODO implement
//     NF_ERROR_EXIT("Not implemented");
//     SpatialOperator result = lhs;
//     // if (!result.getCoefficient().useSpan)
//     // {
//     //     result.setField(std::make_shared<Field<scalar>>(result.exec(), result.nCells(), 1.0));
//     // }
//     // map(result.exec(), result.getCoefficient().values, scaleFunc);
//     return result;
// }

} // namespace dsl
