// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <utility>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/DSL/operator.hpp"
#include "NeoFOAM/core/error.hpp"

namespace NeoFOAM::DSL
{

class EqnSystem
{
public:

    EqnSystem(const NeoFOAM::Executor& exec, std::size_t nCells)
        : exec_(exec), nCells_(nCells), temporalOperators_(), implicitOperators_(),
          explicitOperators_(), volumeField_(nullptr)
    {}

    NeoFOAM::Field<NeoFOAM::scalar> explicitOperation()
    {
        NeoFOAM::Field<NeoFOAM::scalar> source(exec_, nCells_);
        NeoFOAM::fill(source, 0.0);
        for (auto& Operator : explicitOperators_)
        {
            Operator.explicitOperation(source);
        }
        return source;
    }

    void addOperator(const Operator& Operator)
    {
        switch (Operator.getType())
        {
        case Operator::Type::Temporal:
            temporalOperators_.push_back(Operator);
            break;
        case Operator::Type::Implicit:
            implicitOperators_.push_back(Operator);
            break;
        case Operator::Type::Explicit:
            explicitOperators_.push_back(Operator);
            break;
        }
    }

    void addSystem(const EqnSystem& eqnSys)
    {
        for (auto& Operator : eqnSys.temporalOperators_)
        {
            temporalOperators_.push_back(Operator);
        }
        for (auto& Operator : eqnSys.implicitOperators_)
        {
            implicitOperators_.push_back(Operator);
        }
        for (auto& Operator : eqnSys.explicitOperators_)
        {
            explicitOperators_.push_back(Operator);
        }
    }

    void solve()
    {
        if (temporalOperators_.size() == 0 && implicitOperators_.size() == 0)
        {
            NF_ERROR_EXIT("No temporal or implicit terms to solve.");
        }
        if (temporalOperators_.size() > 0)
        {
            // integrate equations in time
        }
        else
        {
            // solve sparse matrix system
        }
    }

    size_t size() const
    {
        return temporalOperators_.size() + implicitOperators_.size() + explicitOperators_.size();
    }

    // getters
    const std::vector<Operator>& temporalOperators() const { return temporalOperators_; }

    const std::vector<Operator>& implicitOperators() const { return implicitOperators_; }

    const std::vector<Operator>& explicitOperators() const { return explicitOperators_; }

    std::vector<Operator>& temporalOperators() { return temporalOperators_; }

    std::vector<Operator>& implicitOperators() { return implicitOperators_; }

    std::vector<Operator>& explicitOperators() { return explicitOperators_; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    const std::size_t nCells() const { return nCells_; }

    scalar getDt() const { return dt_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField()
    {
        if (temporalOperators_.size() == 0 && implicitOperators_.size() == 0)
        {
            NF_ERROR_EXIT("No temporal or implicit terms to solve.");
        }
        if (temporalOperators_.size() > 0)
        {
            // FIXME
            NF_ERROR_EXIT("Not implemented.");
            // volumeField_ = temporalOperators_[0].volumeField();
        }
        else
        {
            // FIXME
            NF_ERROR_EXIT("Not implemented.");
            // volumeField_ = implicitOperators_[0].volumeField();
        }
        return volumeField_;
    }

    NeoFOAM::scalar dt_ = 0;

private:

    const NeoFOAM::Executor exec_;
    const std::size_t nCells_;
    std::vector<Operator> temporalOperators_;
    std::vector<Operator> implicitOperators_;
    std::vector<Operator> explicitOperators_;
    fvcc::VolumeField<NeoFOAM::scalar>* volumeField_;
};

EqnSystem operator+(EqnSystem lhs, const EqnSystem& rhs)
{
    lhs.addSystem(rhs);
    return lhs;
}

EqnSystem operator+(EqnSystem lhs, const Operator& rhs)
{
    lhs.addOperator(rhs);
    return lhs;
}

EqnSystem operator+(const Operator& lhs, const Operator& rhs)
{
    NF_ERROR_EXIT("Not implemented.");
    //     EqnSystem eqnSys(lhs.exec(), lhs.nCells());
    //     eqnSys.addOperator(lhs);
    //     eqnSys.addOperator(rhs);
    //     return eqnSys;
}

EqnSystem operator*(NeoFOAM::scalar scale, const EqnSystem& es)
{
    EqnSystem results(es.exec(), es.nCells());
    for (const auto& Operator : es.temporalOperators())
    {
        results.addOperator(scale * Operator);
    }
    for (const auto& Operator : es.implicitOperators())
    {
        results.addOperator(scale * Operator);
    }
    for (const auto& Operator : es.explicitOperators())
    {
        results.addOperator(scale * Operator);
    }
    return results;
}

EqnSystem operator-(EqnSystem lhs, const EqnSystem& rhs)
{
    lhs.addSystem(-1.0 * rhs);
    return lhs;
}

EqnSystem operator-(EqnSystem lhs, const Operator& rhs)
{
    lhs.addOperator(-1.0 * rhs);
    return lhs;
}

EqnSystem operator-(const Operator& lhs, const Operator& rhs)
{
    NF_ERROR_EXIT("Not implemented.");
    // EqnSystem results(lhs.exec(), lhs.nCells());
    // results.addOperator(lhs);
    // results.addOperator(-1.0 * rhs);
    // return results;
}


} // namespace NeoFOAM::DSL
