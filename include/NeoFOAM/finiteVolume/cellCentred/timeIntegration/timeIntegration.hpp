// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

#include "NeoFOAM/DSL/operator.hpp"
#include "NeoFOAM/DSL/equation.hpp"

#include <functional>

namespace NeoFOAM
{

class TimeIntegrationFactory :
    public RuntimeSelectionFactory<
        TimeIntegrationFactory,
        Parameters<const DSL::Equation&, const Dictionary&>>
{

public:

    static std::string name() { return "timeIntegrationFactory"; }

    TimeIntegrationFactory(const DSL::Equation& eqnSystem, const Dictionary& dict)
        : eqnSystem_(eqnSystem), dict_(dict)
    {}

    virtual ~TimeIntegrationFactory() {}

    virtual void solve() = 0; // Pure virtual function for solving

    // Pure virtual function for cloning
    virtual std::unique_ptr<TimeIntegrationFactory> clone() const = 0;

protected:

    DSL::Equation eqnSystem_;

    const Dictionary& dict_;
};

class TimeIntegration
{

public:

    TimeIntegration(const TimeIntegration& timeIntegrate)
        : timeIntegrateStrategy_(timeIntegrate.timeIntegrateStrategy_->clone()) {};

    TimeIntegration(TimeIntegration&& timeIntegrate)
        : timeIntegrateStrategy_(std::move(timeIntegrate.timeIntegrateStrategy_)) {};

    TimeIntegration(const DSL::Equation& eqnSystem, const Dictionary& dict)
        : timeIntegrateStrategy_(
            TimeIntegrationFactory::create(dict.get<std::string>("type"), eqnSystem, dict)
        ) {};

    void solve() { timeIntegrateStrategy_->solve(); }

private:

    std::unique_ptr<TimeIntegrationFactory> timeIntegrateStrategy_;
};


} // namespace NeoFOAM
