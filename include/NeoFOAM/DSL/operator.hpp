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
#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/core/inputs.hpp"
#include "NeoFOAM/DSL/coeff.hpp"


namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::DSL
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


/* @class OperatorMixin
 * @brief A mixin class to represent simplify implementations of concrete operators
 * in NeoFOAMs DSL
 *
 *
 * @ingroup DSL
 */
class OperatorMixin
{

public:

    OperatorMixin(const Executor exec) : exec_(exec), coeffs_(), evaluated_(false) {};

    // OperatorMixin(const Executor exec, const Field<scalar>* field)
    //     : exec_(exec), field_(field), coeffs_(), evaluated_(false) {};

    virtual ~OperatorMixin() = default;

    // void setField(const Field<scalar>> field)
    // {
    //     field_ = field;
    //     // FIXME
    //     // getCoefficient() = Coeff(field_->span(), getCoefficient().value);
    // }

    const Executor& exec() const { return exec_; }

    Coeff& getCoefficient() { return coeffs_; }

    const Coeff& getCoefficient() const { return coeffs_; }

    bool evaluated() const { return evaluated_; }

    bool& evaluated() { return evaluated_; }

    //
    // const Field<scalar>* field() { return field_; }

protected:

    const Executor exec_; //!< Executor associated with the field. (CPU, GPU, openMP, etc.)

    // NOTE unfortunately does not work
    // std::optional<const Field<scalar>&> field_;

    // const Field<scalar>* field_;

    Coeff coeffs_;

    bool evaluated_;
};


/* @class Operator
 * @brief A class to represent a operator in NeoFOAMs DSL
 *
 * The design here is based on the type erasure design pattern
 * see https://www.youtube.com/watch?v=4eeESJQk-mw
 *
 * Motivation for using type erasure is that concrete implementation
 * of Operators e.g Divergence, Laplacian, etc can be stored in a vector of
 * Operators
 *
 * @ingroup DSL
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

    Operator(const Operator& eqnOperator) : model_ {eqnOperator.model_->clone()} {}

    Operator(Operator&& eqnOperator) : model_ {std::move(eqnOperator.model_)} {}

    Operator& operator=(const Operator& eqnOperator)
    {
        model_ = eqnOperator.model_->clone();
        return *this;
    }

    Operator& operator=(Operator&& eqnOperator)
    {
        model_ = std::move(eqnOperator.model_);
        return *this;
    }

    // std::string display() const { return model_->display(); }

    void explicitOperation(Field<scalar>& source) { model_->explicitOperation(source); }

    void temporalOperation(Field<scalar>& field) { model_->temporalOperation(field); }

    /* returns the fundamental type of an operator, ie explicit, implicit, temporal */
    Operator::Type getType() const { return model_->getType(); }

    void setField(std::shared_ptr<Field<scalar>> field) { model_->setField(field); }

    Coeff& getCoefficient() { return model_->getCoefficient(); }

    Coeff getCoefficient() const { return model_->getCoefficient(); }

    bool evaluated() { return model_->evaluated(); }

    void build(const Input& input) { model_->build(input); }

    const Executor& exec() const { return model_->exec(); }

    // std::size_t nCells() const { return model_->nCells(); }

    // fvcc::VolumeField<scalar>* volumeField() { return model_->volumeField(); }


private:

    /* @brief Base class defining the concept of a term. This effectively
     * defines what functions need to be implemented by a concrete Operator implementation
     *
     *
     * */
    struct OperatorConcept
    {
        virtual ~OperatorConcept() = default;

        virtual std::string display() const = 0;

        virtual void explicitOperation(Field<scalar>& source) = 0;

        virtual void temporalOperation(Field<scalar>& field) = 0;

        virtual void build(const Input& input) = 0;

        virtual bool evaluated() = 0;

        /* returns the fundamental type of an operator, ie explicit, implicit, temporal */
        virtual Operator::Type getType() const = 0;

        virtual void setField(std::shared_ptr<Field<scalar>> field) = 0;

        /* @brief get the associated coefficient for this term */
        virtual Coeff& getCoefficient() = 0;

        /* @brief get the associated coefficient for this term */
        virtual Coeff getCoefficient() const = 0;

        virtual const Executor& exec() const = 0;

        // virtual std::size_t nCells() const = 0;

        // virtual fvcc::VolumeField<scalar>* volumeField() = 0;

        // The Prototype Design Pattern
        virtual std::unique_ptr<OperatorConcept> clone() const = 0;
    };

    // Templated derived class to implement the type-specific behavior
    template<typename T>
    struct OperatorModel : OperatorConcept
    {
        OperatorModel(T cls) : cls_(std::move(cls)) {}

        std::string display() const override
        { /*return cls_.display();*/
        }

        virtual void explicitOperation(Field<scalar>& source) override
        {
            if constexpr (HasExplicitOperator<T>)
            {
                cls_.explicitOperation(source);
            }
        }

        virtual void temporalOperation(Field<scalar>& field) override
        {
            if constexpr (HasTemporalOperator<T>)
            {
                cls_.temporalOperation(field);
            }
        }

        // virtual fvcc::VolumeField<scalar>* volumeField() override { return cls_.volumeField(); }

        virtual void build(const Input& input) override
        {
            // FIXME
            // cls_.build(input);
        }

        virtual bool evaluated() override
        {
            // FIXME
            // return cls_.evaluated();
        }

        /* returns the fundamental type of an operator, ie explicit, implicit, temporal */
        Operator::Type getType() const override { return cls_.getType(); }

        const Executor& exec() const override { return cls_.exec(); }

        // std::size_t nCells() const override { return cls_.nCells(); }

        void setField(std::shared_ptr<Field<scalar>> field) override
        {
            // FIXME
            // cls_.setField(field);
        }

        /* @brief get the associated coefficient for this term */
        virtual Coeff& getCoefficient() override { return cls_.getCoefficient(); }

        /* @brief get the associated coefficient for this term */
        virtual Coeff getCoefficient() const override { return cls_.getCoefficient(); }

        // The Prototype Design Pattern
        std::unique_ptr<OperatorConcept> clone() const override
        {
            return std::make_unique<OperatorModel>(*this);
        }

        T cls_;
    };

    std::unique_ptr<OperatorConcept> model_;
};


auto operator*(scalar scale, const Operator& lhs)
{
    Operator result = lhs;
    result.getCoefficient() *= scale;
    return result;
}

// add multiply operator to Operator
auto operator*(Field<scalar> scale, const Operator& lhs)
{
    Operator result = lhs;
    // FIXME
    // if (!result.getCoefficient().useSpan)
    // {
    //     // allocate the scaling field to avoid deallocation
    //     result.setField(std::make_shared<Field<scalar>>(scale));
    // }
    // else
    // {
    //     result.getCoefficient() *= scale;
    // }
    return result;
}

// template<ForFieldKernel ScaleFunction>
template<typename ScaleFunction>
Operator operator*(ScaleFunction scaleFunc, const Operator& lhs)
{
    Operator result = lhs;
    // FIXME
    // if (!result.getCoefficient().useSpan)
    // {
    //     result.setField(std::make_shared<Field<scalar>>(result.exec(), result.nCells(), 1.0));
    // }
    // map(result.exec(), result.getCoefficient().values, scaleFunc);
    return result;
}

} // namespace NeoFOAM::DSL
