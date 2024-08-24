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

    EqnTermMixin(bool isEvaluated) : scaleField_(), scale_(1.0), termEvaluated(isEvaluated) {};

    virtual ~EqnTermMixin() = default;

    std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>>& scaleField() { return scaleField_; }

    const std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>> scaleField() const
    {
        return scaleField_;
    }

    NeoFOAM::scalar& scaleValue() { return scale_; }

    NeoFOAM::scalar scaleValue() const { return scale_; }

    std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>> scaleField_;
    NeoFOAM::scalar scale_ = 1.0;

    bool termEvaluated() { return termEvaluated; }

protected:

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

    void setScale(NeoFOAM::scalar scale) { model_->scaleValue() = scale; }

    NeoFOAM::scalar& scaleValue() { return model_->scaleValue(); }

    NeoFOAM::scalar scaleValue() const { return model_->scaleValue(); }

    std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>>& scaleField() { return model_->scaleField(); }

    std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>> scaleField() const
    {
        return model_->scaleField;
    }

    bool termEvaluated() { return model_->termEvaluated(); }

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

        virtual bool termEvaluated() = 0;

        virtual EqnTerm::Type getType() const = 0;
        virtual NeoFOAM::scalar& scaleValue() = 0;
        virtual NeoFOAM::scalar scaleValue() const = 0;

        virtual std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>>& scaleField() = 0;
        virtual std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>> scaleField() const = 0;

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

        virtual bool termEvaluated() override { return cls_.termEvaluated(); }

        EqnTerm::Type getType() const override { return cls_.getType(); }

        const NeoFOAM::Executor& exec() const override { return cls_.exec(); }

        std::size_t nCells() const override { return cls_.nCells(); }

        std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>>& scaleField() override
        {
            return cls_.scaleField();
        }

        std::shared_ptr<NeoFOAM::Field<NeoFOAM::scalar>> scaleField() const override
        {
            return cls_.scaleField();
        }

        NeoFOAM::scalar& scaleValue() override { return cls_.scaleValue(); }

        NeoFOAM::scalar scaleValue() const override { return cls_.scaleValue(); }

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
    result.scaleValue() *= scale;
    return result;
}

// add multiply operator to EqnTerm
template<typename ValueType>
EqnTerm<ValueType> operator*(NeoFOAM::Field<NeoFOAM::scalar> scale, const EqnTerm<ValueType>& lhs)
{
    EqnTerm<ValueType> result = lhs;
    if (!result.scaleField())
    {
        result.scaleField() = std::make_shared<NeoFOAM::Field<NeoFOAM::scalar>>(scale);
    }
    else
    {
        auto sRes = result.scaleField()->span();
        auto sSpan = scale.span();
        result.scaleField()->apply(KOKKOS_LAMBDA(const NeoFOAM::size_t i) {
            return sRes[i] * sSpan[i];
        });
    }
    return result;
}

template<typename ValueType, parallelForFieldKernel<ValueType> ScaleFunction>
EqnTerm<ValueType> operator*(ScaleFunction scaleFunc, const EqnTerm<ValueType>& lhs)
{
    EqnTerm<ValueType> result = lhs;
    if (!result.scaleField())
    {
        result.scaleField() =
            std::make_shared<NeoFOAM::Field<NeoFOAM::scalar>>(lhs.exec(), lhs.nCells());
    }
    NeoFOAM::map(*result.scaleField(), scaleFunc);
    return result;
}

} // namespace NeoFOAM::DSL
