// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <memory>
#include <concepts>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/coeff.hpp"

namespace la = NeoFOAM::la;

namespace NeoFOAM::dsl
{

template<typename T>
concept HasExplicitOperator = requires(T t) {
    {
        t.explicitOperation(std::declval<Field<scalar>&>())
    } -> std::same_as<void>; // Adjust return type and arguments as needed
};

template<typename T>
concept HasImplicitOperator = requires(T t) {
    {
        t.implicitOperation(std::declval<la::LinearSystem<scalar, localIdx>&>())
    } -> std::same_as<void>; // Adjust return type and arguments as needed
};

template<typename T>
concept IsSpatialOperator = HasExplicitOperator<T> || HasImplicitOperator<T>;

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
class SpatialOperator
{
public:

    enum class Type
    {
        Temporal,
        Implicit,
        Explicit
    };

    template<IsSpatialOperator T>
    SpatialOperator(T cls) : model_(std::make_unique<OperatorModel<T>>(std::move(cls)))
    {}

    SpatialOperator(const SpatialOperator& eqnOperator);

    SpatialOperator(SpatialOperator&& eqnOperator);

    void explicitOperation(Field<scalar>& source);

    void implicitOperation(la::LinearSystem<scalar, localIdx>& ls);

    la::LinearSystem<scalar, localIdx> createEmptyLinearSystem() const;

    /* returns the fundamental type of an operator, ie explicit, implicit */
    SpatialOperator::Type getType() const;

    std::string getName() const;

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

        virtual void implicitOperation(la::LinearSystem<scalar, localIdx>& ls) = 0;

        virtual la::LinearSystem<scalar, localIdx> createEmptyLinearSystem() const = 0;

        /* @brief Given an input this function reads required coeffs */
        virtual void build(const Input& input) = 0;

        /* returns the name of the operator */
        virtual std::string getName() const = 0;

        /* returns the fundamental type of an operator, ie explicit, implicit */
        virtual SpatialOperator::Type getType() const = 0;

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

        virtual void implicitOperation(la::LinearSystem<scalar, localIdx>& ls) override
        {
            if constexpr (HasImplicitOperator<ConcreteOperatorType>)
            {
                concreteOp_.implicitOperation(ls);
            }
        }

        virtual la::LinearSystem<scalar, localIdx> createEmptyLinearSystem() const override
        {
            if constexpr (HasImplicitOperator<ConcreteOperatorType>)
            {
                return concreteOp_.createEmptyLinearSystem();
            }
            throw std::runtime_error("Implicit operation not implemented");
            // only need to avoid compiler warning about missing return statement
            // this code path should never be reached as we call implicitOperation on an explicit
            // operator
            NeoFOAM::Field<NeoFOAM::scalar> values(exec(), 1, 0.0);
            NeoFOAM::Field<NeoFOAM::localIdx> colIdx(exec(), 1, 0);
            NeoFOAM::Field<NeoFOAM::localIdx> rowPtrs(exec(), 2, 0);
            NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> csrMatrix(
                values, colIdx, rowPtrs
            );

            NeoFOAM::Field<NeoFOAM::scalar> rhs(exec(), 1, 0.0);
            return la::LinearSystem<scalar, localIdx>(csrMatrix, rhs, "custom");
        }

        /* @brief Given an input this function reads required coeffs */
        virtual void build(const Input& input) override { concreteOp_.build(input); }

        /* returns the fundamental type of an operator, ie explicit, implicit, temporal */
        SpatialOperator::Type getType() const override { return concreteOp_.getType(); }

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


SpatialOperator operator*(scalar scalarCoeff, SpatialOperator rhs);

SpatialOperator operator*(const Field<scalar>& coeffField, SpatialOperator rhs);

SpatialOperator operator*(const Coeff& coeff, SpatialOperator rhs);

template<typename CoeffFunction>
    requires std::invocable<CoeffFunction&, size_t>
SpatialOperator operator*([[maybe_unused]] CoeffFunction coeffFunc, const SpatialOperator& lhs)
{
    // TODO implement
    NF_ERROR_EXIT("Not implemented");
    SpatialOperator result = lhs;
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

    OperatorMixin(const Executor exec, FieldType& field, SpatialOperator::Type type)
        : exec_(exec), coeffs_(), field_(field), type_(type) {};

    SpatialOperator::Type getType() const { return type_; }

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

    SpatialOperator::Type type_;
};
} // namespace NeoFOAM::dsl
