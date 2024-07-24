// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#include "NeoFOAM/core/primitives/scalar.hpp"

namespace NeoFOAM::DSL
{

// EqnTerm class that uses type erasure without inheritance
class EqnTerm
{
public:

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

    void explicitOperation(NeoFOAM::scalar& exp)
    {
        model_->explicitOperation(exp, model_->scaleCoeff);
    }

    void setScale(NeoFOAM::scalar scale) { model_->scaleCoeff *= scale; }


private:

    // Base class to hold the type-erased value and the display function
    struct Concept
    {
        virtual ~Concept() = default;
        virtual std::string display() const = 0;
        virtual void explicitOperation(NeoFOAM::scalar& exp, NeoFOAM::scalar scale) = 0;
        // The Prototype Design Pattern
        virtual std::unique_ptr<Concept> clone() const = 0;
        NeoFOAM::scalar scaleCoeff = 1.0;
    };

    // Templated derived class to implement the type-specific behavior
    template<typename T>
    struct Model : Concept
    {
        Model(T cls) : cls_(std::move(cls)) {}

        std::string display() const override { return cls_.display(); }

        virtual void explicitOperation(NeoFOAM::scalar& exp, NeoFOAM::scalar scale) override
        {
            cls_.explicitOperation(exp, scale);
        }

        T cls_;

        // The Prototype Design Pattern
        std::unique_ptr<Concept> clone() const override { return std::make_unique<Model>(*this); }
    };

    std::unique_ptr<Concept> model_;
};

} // namespace NeoFOAM::DSL
