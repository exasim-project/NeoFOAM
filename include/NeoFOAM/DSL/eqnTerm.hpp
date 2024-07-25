// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"

namespace NeoFOAM::DSL
{

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

    EqnTerm::Type getType() const { return model_->getType(); }

    void setScale(NeoFOAM::scalar scale) { model_->scaleCoeff *= scale; }


    const NeoFOAM::Executor& exec() const { return model_->exec(); }

    std::size_t nCells() const { return model_->nCells(); }


private:

    // Base class to hold the type-erased value and the display function
    struct Concept
    {
        virtual ~Concept() = default;
        virtual std::string display() const = 0;
        virtual void
        explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source, NeoFOAM::scalar scale) = 0;
        NeoFOAM::scalar scaleCoeff = 1.0;
        virtual EqnTerm::Type getType() const = 0;

        virtual const NeoFOAM::Executor& exec() const = 0;
        virtual std::size_t nCells() const = 0;

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
            cls_.explicitOperation(source, scale);
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

// EqnTerm operator*(const EqnTerm& lhs,NeoFOAM::scalar scale)
// {
//     EqnTerm result = lhs;
//     result.setScale(scale);
//     return result;
// }

} // namespace NeoFOAM::DSL
