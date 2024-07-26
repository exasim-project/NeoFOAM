// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <utility>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/core/error.hpp"

namespace NeoFOAM::DSL
{

class EqnSystem
{
public:

    EqnSystem(const NeoFOAM::Executor& exec, std::size_t nCells)
        : exec_(exec), nCells_(nCells), temporalTerms_(), implicitTerms_(), explicitTerms_(), volumeField_(nullptr)
    {}

    NeoFOAM::Field<NeoFOAM::scalar> explicitOperation()
    {
        NeoFOAM::Field<NeoFOAM::scalar> source(exec_, nCells_);
        NeoFOAM::fill(source, 0.0);
        for (auto& eqnTerm : explicitTerms_)
        {
            eqnTerm.explicitOperation(source);
        }
        return source;
    }

    void addTerm(const EqnTerm& eqnTerm)
    {
        switch (eqnTerm.getType())
        {
        case EqnTerm::Type::Temporal:
            temporalTerms_.push_back(eqnTerm);
            break;
        case EqnTerm::Type::Implicit:
            implicitTerms_.push_back(eqnTerm);
            break;
        case EqnTerm::Type::Explicit:
            explicitTerms_.push_back(eqnTerm);
            break;
        }
    }

    void addSystem(const EqnSystem& eqnSys)
    {
        for (auto& eqnTerm : eqnSys.temporalTerms_)
        {
            temporalTerms_.push_back(eqnTerm);
        }
        for (auto& eqnTerm : eqnSys.implicitTerms_)
        {
            implicitTerms_.push_back(eqnTerm);
        }
        for (auto& eqnTerm : eqnSys.explicitTerms_)
        {
            explicitTerms_.push_back(eqnTerm);
        }
    }

    void solve()
    {
        if (temporalTerms_.size() == 0 && implicitTerms_.size() == 0)
        {
            NF_ERROR_EXIT("No temporal or implicit terms to solve.");
        }
        if (temporalTerms_.size() > 0)
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
        return temporalTerms_.size() + implicitTerms_.size() + explicitTerms_.size();
    }

    // getters
    const std::vector<EqnTerm>& temporalTerms() const { return temporalTerms_; }

    const std::vector<EqnTerm>& implicitTerms() const { return implicitTerms_; }

    const std::vector<EqnTerm>& explicitTerms() const { return explicitTerms_; }

    std::vector<EqnTerm>& temporalTerms() { return temporalTerms_; }

    std::vector<EqnTerm>& implicitTerms() { return implicitTerms_; }

    std::vector<EqnTerm>& explicitTerms() { return explicitTerms_; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    const std::size_t nCells() const { return nCells_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() 
    {
        if (temporalTerms_.size() == 0 && implicitTerms_.size() == 0)
        {
            NF_ERROR_EXIT("No temporal or implicit terms to solve.");
        }
        if (temporalTerms_.size() > 0)
        {
            volumeField_ = temporalTerms_[0].volumeField();
        }
        else
        {
            volumeField_ = implicitTerms_[0].volumeField();
        }
        return volumeField_; 
    }

private:

    const NeoFOAM::Executor exec_;
    const std::size_t nCells_;
    std::vector<EqnTerm> temporalTerms_;
    std::vector<EqnTerm> implicitTerms_;
    std::vector<EqnTerm> explicitTerms_;
    fvcc::VolumeField<NeoFOAM::scalar>* volumeField_;
};

EqnSystem operator+(const EqnSystem& lhs, const EqnSystem& rhs)
{
    std::cout << "Adding EqnSystem from EqnSystem" << std::endl;
    EqnSystem results = lhs;
    results.addSystem(rhs);
    return results;
}

EqnSystem operator+(const EqnSystem& lhs, const EqnTerm& rhs)
{
    EqnSystem results = lhs;
    results.addTerm(rhs);
    return results;
}

EqnSystem operator+(const EqnTerm& lhs, const EqnTerm& rhs)
{
    EqnSystem eqnSys(lhs.exec(), lhs.nCells());
    eqnSys.addTerm(lhs);
    eqnSys.addTerm(rhs);
    return eqnSys;
}

EqnSystem operator*(NeoFOAM::scalar scale, const EqnSystem& es)
{
    EqnSystem results(es.exec(), es.nCells());
    for (const auto& eqnTerm : es.temporalTerms())
    {
        results.addTerm(scale * eqnTerm);
    }
    for (const auto& eqnTerm : es.implicitTerms())
    {
        results.addTerm(scale * eqnTerm);
    }
    for (const auto& eqnTerm : es.explicitTerms())
    {
        results.addTerm(scale * eqnTerm);
    }
    return results;
}

EqnSystem operator-(const EqnSystem& lhs, const EqnSystem& rhs)
{
    EqnSystem results = lhs;
    results.addSystem(-1.0 * rhs);
    return results;
}

EqnSystem operator-(const EqnSystem& lhs, const EqnTerm& rhs)
{
    EqnSystem results = lhs;
    results.addTerm(-1.0 * rhs);
    return results;
}

EqnSystem operator-(const EqnTerm& lhs, const EqnTerm& rhs)
{
    EqnSystem results(lhs.exec(), lhs.nCells());
    results.addTerm(lhs);
    results.addTerm(-1.0 * rhs);
    return results;
}


} // namespace NeoFOAM::DSL
