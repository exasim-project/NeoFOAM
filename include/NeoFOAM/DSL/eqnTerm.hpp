// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::DSL
{

template<typename T>
concept HasTemporalTerm = requires(T t) {
    {
        t.temporalOperation(
            std::declval<NeoFOAM::Field<NeoFOAM::scalar>&>(), std::declval<NeoFOAM::scalar>()
        )
    } -> std::same_as<void>; // Adjust return type and arguments as needed
};

template<typename T>
concept HasExplicitTerm = requires(T t) {
    {
        t.explicitOperation(
            std::declval<NeoFOAM::Field<NeoFOAM::scalar>&>(), std::declval<NeoFOAM::scalar>()
        )
    } -> std::same_as<void>; // Adjust return type and arguments as needed
};
// EqnTerm class that uses type erasure without inheritance
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

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source)
    {
        model_->explicitOperation(source, model_->scaleCoeff);
    }

    void temporalOperation(NeoFOAM::Field<NeoFOAM::scalar>& field)
    {
        model_->temporalOperation(field, model_->scaleCoeff);
    }

    EqnTerm::Type getType() const { return model_->getType(); }

    void setScale(NeoFOAM::scalar scale) { model_->scaleCoeff *= scale; }


    const NeoFOAM::Executor& exec() const { return model_->exec(); }

    std::size_t nCells() const { return model_->nCells(); }


    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() { return model_->volumeField(); }


private:

    // Base class to hold the type-erased value and the display function
    struct Concept
    {
        virtual ~Concept() = default;
        virtual std::string display() const = 0;
        virtual void
        explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source, NeoFOAM::scalar scale) = 0;
        virtual void
        temporalOperation(NeoFOAM::Field<NeoFOAM::scalar>& field, NeoFOAM::scalar scale) = 0;
        NeoFOAM::scalar scaleCoeff = 1.0;
        virtual EqnTerm::Type getType() const = 0;

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

        virtual void
        explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source, NeoFOAM::scalar scale) override
        {
            if constexpr (HasExplicitTerm<T>)
            {
                cls_.explicitOperation(source, scale);
            }
        }

        virtual void
        temporalOperation(NeoFOAM::Field<NeoFOAM::scalar>& field, NeoFOAM::scalar scale) override
        {
            if constexpr (HasTemporalTerm<T>)
            {
                cls_.temporalOperation(field, scale);
            }
        }

        virtual fvcc::VolumeField<NeoFOAM::scalar>* volumeField() override
        {
            return cls_.volumeField();
        }

        EqnTerm::Type getType() const override { return cls_.getType(); }

        const NeoFOAM::Executor& exec() const override { return cls_.exec(); }

        std::size_t nCells() const override { return cls_.nCells(); }

        // The Prototype Design Pattern
        std::unique_ptr<Concept> clone() const override { return std::make_unique<Model>(*this); }

        T cls_;
    };

    std::unique_ptr<Concept> model_;
};


// add multiply operator to EqnTerm
EqnTerm operator*(NeoFOAM::scalar scale, const EqnTerm& lhs)
{
    EqnTerm result = lhs;
    result.setScale(scale);
    return result;
}

} // namespace NeoFOAM::DSL
