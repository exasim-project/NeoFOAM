// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <utility>

#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/core/error.hpp"

namespace NeoFOAM::DSL
{

class EqnSystem
{
public:

    NeoFOAM::scalar explicitOperation()
    {
        NeoFOAM::scalar exp = 0;
        for (auto& eqnTerm : explicitTerms_)
        {
            eqnTerm.explicitOperation(exp);
        }
        return exp;
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

private:

    std::vector<EqnTerm> temporalTerms_;
    std::vector<EqnTerm> implicitTerms_;
    std::vector<EqnTerm> explicitTerms_;
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
    EqnSystem eqnSys;
    eqnSys.addTerm(lhs);
    eqnSys.addTerm(rhs);
    return eqnSys;
}

EqnSystem operator*(NeoFOAM::scalar scale, const EqnSystem& es)
{
    EqnSystem results;
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
    std::cout << "Subtracting EqnSystem from EqnSystem" << std::endl;
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
    EqnSystem results;
    results.addTerm(lhs);
    results.addTerm(-1.0 * rhs);
    return results;
}


} // namespace NeoFOAM::DSL
