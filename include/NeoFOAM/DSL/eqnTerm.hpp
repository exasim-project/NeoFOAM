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


namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::DSL
{

template<typename T>
concept HasTemporalTerm = requires(T t) {
    {
        t.temporalOperation(std::declval<NeoFOAM::Field<NeoFOAM::scalar>&>())
    } -> std::same_as<void>; // Adjust return type and arguments as needed
};

template<typename T>
concept HasExplicitTerm = requires(T t) {
    {
        t.explicitOperation(std::declval<NeoFOAM::Field<NeoFOAM::scalar>&>())
    } -> std::same_as<void>; // Adjust return type and arguments as needed
};


template<typename ValueType>
class EqnTermMixin
{
public:

    EqnTermMixin(bool isEvaluated) : field_(), scaleField_(1.0), termEvaluated(isEvaluated) {};

    virtual ~EqnTermMixin() = default;

    void setField(std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>> field)
    {
        field_ = field;
        scaleField() = ScalingField<ValueType>(field_->span(), scaleField().value);
    }

    NeoFOAM::ScalingField<ValueType>& scaleField() { return scaleField_; }

    NeoFOAM::ScalingField<ValueType> scaleField() const { return scaleField_; }

    bool evaluated() { return termEvaluated; }

    std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>>& field() { return field_; }

protected:

    // used to allocate the shared pointer scaling field
    // requires a span and the field is required to keep the pointer alive

    std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>> field_;

    NeoFOAM::ScalingField<ValueType> scaleField_;

    bool termEvaluated;
};


template<typename ValueType>
class EqnTerm
{
public:

    enum class Type
    {
        Temporal,
        Implicit,
        Explicit
    };

    template<typename T>
    EqnTerm(T cls) : model_(std::make_unique<Model<T>>(std::move(cls)))
    {}

    EqnTerm(const EqnTerm& eqnTerm) : model_ {eqnTerm.model_->clone()} {}

    EqnTerm(EqnTerm&& eqnTerm) : model_ {std::move(eqnTerm.model_)} {}

    EqnTerm& operator=(const EqnTerm& eqnTerm)
    {
        model_ = eqnTerm.model_->clone();
        return *this;
    }

    EqnTerm& operator=(EqnTerm&& eqnTerm)
    {
        model_ = std::move(eqnTerm.model_);
        return *this;
    }

    std::string display() const { return model_->display(); }

    void explicitOperation(NeoFOAM::Field<ValueType>& source) { model_->explicitOperation(source); }

    void temporalOperation(NeoFOAM::Field<ValueType>& field) { model_->temporalOperation(field); }

    EqnTerm::Type getType() const { return model_->getType(); }

    void setField(std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>> field)
    {
        model_->setField(field);
    }

    NeoFOAM::ScalingField<ValueType>& scaleField() { return model_->scaleField(); }

    NeoFOAM::ScalingField<ValueType> scaleField() const { return model_->scaleField(); }

    bool evaluated() { return model_->evaluated(); }

    void build(const NeoFOAM::Input& input) { model_->build(input); }

    const NeoFOAM::Executor& exec() const { return model_->exec(); }

    std::size_t nCells() const { return model_->nCells(); }


    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() { return model_->volumeField(); }


private:

    // Base class to hold the type-erased value and the display function
    struct Concept
    {
        virtual ~Concept() = default;
        virtual std::string display() const = 0;
        virtual void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source) = 0;
        virtual void temporalOperation(NeoFOAM::Field<NeoFOAM::scalar>& field) = 0;
        virtual void build(const NeoFOAM::Input& input) = 0;

        virtual bool evaluated() = 0;

        virtual EqnTerm::Type getType() const = 0;

        virtual void setField(std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>> field) = 0;
        virtual NeoFOAM::ScalingField<NeoFOAM::scalar>& scaleField() = 0;
        virtual NeoFOAM::ScalingField<NeoFOAM::scalar> scaleField() const = 0;

        virtual const NeoFOAM::Executor& exec() const = 0;
        virtual std::size_t nCells() const = 0;
        virtual fvcc::VolumeField<NeoFOAM::scalar>* volumeField() = 0;

        // The Prototype Design Pattern
        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    // Templated derived class to implement the type-specific behavior
    template<typename T>
    struct Model : Concept
    {
        Model(T cls) : cls_(std::move(cls)) {}

        std::string display() const override { return cls_.display(); }

        virtual void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source) override
        {
            if constexpr (HasExplicitTerm<T>)
            {
                cls_.explicitOperation(source);
            }
        }

        virtual void temporalOperation(NeoFOAM::Field<NeoFOAM::scalar>& field) override
        {
            if constexpr (HasTemporalTerm<T>)
            {
                cls_.temporalOperation(field);
            }
        }

        virtual fvcc::VolumeField<NeoFOAM::scalar>* volumeField() override
        {
            return cls_.volumeField();
        }

        virtual void build(const NeoFOAM::Input& input) override { cls_.build(input); }

        virtual bool evaluated() override { return cls_.evaluated(); }

        EqnTerm::Type getType() const override { return cls_.getType(); }

        const NeoFOAM::Executor& exec() const override { return cls_.exec(); }

        std::size_t nCells() const override { return cls_.nCells(); }

        void setField(std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>> field) override
        {
            cls_.setField(field);
        }

        virtual NeoFOAM::ScalingField<NeoFOAM::scalar>& scaleField() override
        {
            return cls_.scaleField();
        }

        virtual NeoFOAM::ScalingField<NeoFOAM::scalar> scaleField() const override
        {
            return cls_.scaleField();
        }

        // The Prototype Design Pattern
        std::unique_ptr<Concept> clone() const override { return std::make_unique<Model>(*this); }

        T cls_;
    };

    std::unique_ptr<Concept> model_;
};


// add multiply operator to EqnTerm
template<typename ValueType>
EqnTerm<ValueType> operator*(NeoFOAM::scalar scale, const EqnTerm<ValueType>& lhs)
{
    EqnTerm<ValueType> result = lhs;
    result.scaleField() *= scale;
    return result;
}

// add multiply operator to EqnTerm
template<typename ValueType>
EqnTerm<ValueType> operator*(NeoFOAM::Field<NeoFOAM::scalar> scale, const EqnTerm<ValueType>& lhs)
{
    EqnTerm<ValueType> result = lhs;
    if (!result.scaleField().useSpan)
    {
        // allocate the scaling field to avoid deallocation
        result.setField(std::make_shared<NeoFOAM::Field<NeoFOAM::scalar>>(scale));
    }
    else
    {
        result.scaleField() *= scale;
    }
    return result;
}

template<typename ValueType, parallelForFieldKernel<ValueType> ScaleFunction>
EqnTerm<ValueType> operator*(ScaleFunction scaleFunc, const EqnTerm<ValueType>& lhs)
{
    EqnTerm<ValueType> result = lhs;
    if (!result.scaleField().useSpan)
    {
        result.setField(
            std::make_shared<NeoFOAM::Field<NeoFOAM::scalar>>(result.exec(), result.nCells(), 1.0)
        );
    }
    NeoFOAM::map(result.exec(), result.scaleField().values, scaleFunc);
    return result;
}

} // namespace NeoFOAM::DSL
