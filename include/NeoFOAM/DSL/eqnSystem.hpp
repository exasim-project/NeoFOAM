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
#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/timeIntegration.hpp"


namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::finiteVolume::cellCentred
{
template<typename ValueType>
class TimeIntegration;
}


namespace NeoFOAM::DSL
{

template<typename ValueType>
class EqnSystem
{
public:

    EqnSystem(const NeoFOAM::Executor& exec, std::size_t nCells)
        : exec_(exec), nCells_(nCells), termsEvaluated(false), temporalTerms_(), implicitTerms_(),
          explicitTerms_(), volumeField_(nullptr)
    {}

    void build(NeoFOAM::Input input)
    {
        for (auto& eqnTerm : temporalTerms_)
        {
            eqnTerm.build(input);
        }
        for (auto& eqnTerm : implicitTerms_)
        {
            eqnTerm.build(input);
        }
        for (auto& eqnTerm : explicitTerms_)
        {
            eqnTerm.build(input);
        }
    }

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

    void addTerm(const EqnTerm<ValueType>& eqnTerm)
    {
        switch (eqnTerm.getType())
        {
        case EqnTerm<ValueType>::Type::Temporal:
            temporalTerms_.push_back(eqnTerm);
            break;
        case EqnTerm<ValueType>::Type::Implicit:
            implicitTerms_.push_back(eqnTerm);
            break;
        case EqnTerm<ValueType>::Type::Explicit:
            explicitTerms_.push_back(eqnTerm);
            break;
        }
    }

    void addSystem(const EqnSystem<ValueType>& eqnSys)
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
        bool allTermsEvaluated = evaluated();
        if (!allTermsEvaluated)
        {
            if (fvSchemesDict.empty())
            {
                NF_ERROR_EXIT("No scheme dictionary provided.");
            }
            build(fvSchemesDict);
        }

        if (temporalTerms_.size() == 0 && implicitTerms_.size() == 0)
        {
            NF_ERROR_EXIT("No temporal or implicit terms to solve.");
        }
        if (temporalTerms_.size() > 0)
        {
            // integrate equations in time
            fvcc::TimeIntegration<ValueType> timeIntergrator(
                *this, fvSchemesDict.subDict("ddtSchemes")
            );
            timeIntergrator.solve();
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
    const std::vector<EqnTerm<ValueType>>& temporalTerms() const { return temporalTerms_; }

    const std::vector<EqnTerm<ValueType>>& implicitTerms() const { return implicitTerms_; }

    const std::vector<EqnTerm<ValueType>>& explicitTerms() const { return explicitTerms_; }

    std::vector<EqnTerm<ValueType>>& temporalTerms() { return temporalTerms_; }

    std::vector<EqnTerm<ValueType>>& implicitTerms() { return implicitTerms_; }

    std::vector<EqnTerm<ValueType>>& explicitTerms() { return explicitTerms_; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

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

    NeoFOAM::scalar dt = 0;
    NeoFOAM::Dictionary fvSchemesDict;

private:

    bool evaluated()
    {
        // check if terms have been evaluated
        for (auto& eqnTerm : temporalTerms_)
        {
            if (!eqnTerm.evaluated())
            {
                return false;
            }
        }

        for (auto& eqnTerm : implicitTerms_)
        {
            if (!eqnTerm.evaluated())
            {
                return false;
            }
        }

        for (auto& eqnTerm : explicitTerms_)
        {
            if (!eqnTerm.evaluated())
            {
                return false;
            }
        }

        return true;
    }

    NeoFOAM::Executor exec_;
    std::size_t nCells_;
    bool termsEvaluated = false;
    std::vector<EqnTerm<ValueType>> temporalTerms_;
    std::vector<EqnTerm<ValueType>> implicitTerms_;
    std::vector<EqnTerm<ValueType>> explicitTerms_;
    fvcc::VolumeField<NeoFOAM::scalar>* volumeField_;
};

template<typename ValueType>
EqnSystem<ValueType> operator+(EqnSystem<ValueType> lhs, const EqnSystem<ValueType>& rhs)
{
    lhs.addSystem(rhs);
    return lhs;
}

template<typename ValueType>
EqnSystem<ValueType> operator+(EqnSystem<ValueType> lhs, const EqnTerm<ValueType>& rhs)
{
    lhs.addTerm(rhs);
    return lhs;
}

template<typename ValueType>
EqnSystem<ValueType> operator+(EqnTerm<ValueType> lhs, EqnTerm<ValueType> rhs)
{
    EqnSystem<ValueType> eqnSys(lhs.exec(), lhs.nCells());
    eqnSys.addTerm(lhs);
    eqnSys.addTerm(rhs);
    return eqnSys;
}

template<typename ValueType>
EqnSystem<ValueType> operator*(NeoFOAM::scalar scale, const EqnSystem<ValueType>& es)
{
    EqnSystem<ValueType> results(es.exec(), es.nCells());
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

template<typename ValueType>
EqnSystem<ValueType> operator-(EqnSystem<ValueType> lhs, const EqnSystem<ValueType>& rhs)
{
    lhs.addSystem(-1.0 * rhs);
    return lhs;
}

template<typename ValueType>
EqnSystem<ValueType> operator-(EqnSystem<ValueType> lhs, const EqnTerm<ValueType>& rhs)
{
    lhs.addTerm(-1.0 * rhs);
    return lhs;
}

template<typename ValueType>
EqnSystem<ValueType> operator-(const EqnTerm<ValueType>& lhs, const EqnTerm<ValueType>& rhs)
{
    EqnSystem<ValueType> results(lhs.exec(), lhs.nCells());
    results.addTerm(lhs);
    results.addTerm(-1.0 * rhs);
    return results;
}


} // namespace NeoFOAM::DSL
