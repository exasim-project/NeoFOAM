// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/DSL/eqnSystem.hpp"

#include <functional>

namespace dsl = NeoFOAM::DSL;


namespace NeoFOAM::DSL
{
template<typename ValueType>
class EqnSystem;
}

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
class TimeIntegrationFactory :
    public NeoFOAM::RuntimeSelectionFactory<
        TimeIntegrationFactory<ValueType>,
        Parameters<const dsl::EqnSystem<ValueType>&, const Dictionary&>>
{

public:

    static std::string name() { return "timeIntegrationFactory"; }

    TimeIntegrationFactory(const dsl::EqnSystem<ValueType>& eqnSystem, const Dictionary& dict)
        : eqnSystem_(eqnSystem), dict_(dict)
    {}

    virtual ~TimeIntegrationFactory() {} // Virtual destructor

    virtual void solve() = 0; // Pure virtual function for solving

    // Pure virtual function for cloning
    virtual std::unique_ptr<TimeIntegrationFactory> clone() const = 0;

protected:

    dsl::EqnSystem<ValueType> eqnSystem_;
    const Dictionary& dict_;
};

template<typename ValueType>
class TimeIntegration
{

public:

    TimeIntegration(const TimeIntegration& timeIntegrate)
        : timeIntegrateStrategy_(timeIntegrate.timeIntegrateStrategy_->clone()) {};

    TimeIntegration(TimeIntegration&& timeIntegrate)
        : timeIntegrateStrategy_(std::move(timeIntegrate.timeIntegrateStrategy_)) {};

    TimeIntegration(const dsl::EqnSystem<ValueType>& eqnSystem, const Dictionary& dict)
        : timeIntegrateStrategy_(TimeIntegrationFactory<ValueType>::create(
            dict.get<std::string>("type"), eqnSystem, dict
        )) {};


    void solve() { timeIntegrateStrategy_->solve(); }

private:


    std::unique_ptr<TimeIntegrationFactory<ValueType>> timeIntegrateStrategy_;
};


} // namespace NeoFOAM
