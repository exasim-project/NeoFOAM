// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <utility>

#include "NeoFOAM/DSL/eqnTerm.hpp"

namespace NeoFOAM::DSL
{

class EqnSystem
{
public:

    NeoFOAM::scalar explicitOperation()
    {
        NeoFOAM::scalar exp = 0;
        for (auto& eqnTerm : eqnTerms_)
        {
            eqnTerm.explicitOperation(exp);
        }
        return exp;
    }

    std::vector<EqnTerm> eqnTerms_;
};

EqnSystem operator+(EqnSystem lhs, const EqnSystem& rhs)
{
    for (auto& eqnTerm : rhs.eqnTerms_)
    {
        lhs.eqnTerms_.push_back(eqnTerm);
    }
    return lhs;
}

EqnSystem operator+(EqnSystem lhs, const EqnTerm& rhs)
{
    lhs.eqnTerms_.push_back(rhs);
    return lhs;
}

EqnSystem operator+(const EqnTerm& lhs, const EqnTerm& rhs)
{
    EqnSystem eqnSys;
    eqnSys.eqnTerms_.push_back(lhs);
    eqnSys.eqnTerms_.push_back(rhs);
    return eqnSys;
}

EqnSystem operator-(EqnSystem lhs, EqnSystem rhs)
{
    for (auto& eqnTerm : rhs.eqnTerms_)
    {
        eqnTerm.setScale(-1.0);
        lhs.eqnTerms_.push_back(eqnTerm);
    }
    return lhs;
}

EqnSystem operator-(EqnSystem lhs, EqnTerm rhs)
{
    rhs.setScale(-1.0);
    lhs.eqnTerms_.push_back(rhs);
    return lhs;
}

EqnSystem operator-(EqnTerm lhs, EqnTerm rhs)
{
    EqnSystem eqnSys;
    rhs.setScale(-1.0);
    eqnSys.eqnTerms_.push_back(lhs);
    eqnSys.eqnTerms_.push_back(rhs);
    return eqnSys;
}


} // namespace NeoFOAM::DSL
