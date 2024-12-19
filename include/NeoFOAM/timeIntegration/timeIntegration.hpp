// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/dsl/expression.hpp"

namespace NeoFOAM::timeIntegration
{

/* @class Factory class to create time integration method by a given name
 * using NeoFOAMs runTimeFactory mechanism
 */
template<typename SolutionType>
class TimeIntegratorBase :
    public RuntimeSelectionFactory<TimeIntegratorBase<SolutionType>, Parameters<const Dictionary&>>
{

public:

    using Expression = NeoFOAM::dsl::Expression;

    static std::string name() { return "timeIntegrationFactory"; }

    TimeIntegratorBase(const Dictionary& dict) : dict_(dict) {}

    virtual ~TimeIntegratorBase() {}

    virtual void solve(
        Expression& eqn, SolutionType& sol, scalar t, scalar dt
    ) = 0; // Pure virtual function for solving

    // Pure virtual function for cloning
    virtual std::unique_ptr<TimeIntegratorBase> clone() const = 0;

protected:

    const Dictionary& dict_;
};

/**
 * @class Factory class to create time integration method by a given name
 * using NeoFOAMs runTimeFactory mechanism
 *
 * @tparam SolutionFieldType Type of the solution field eg, volumeField or just a plain Field
 */
template<typename SolutionFieldType>
class TimeIntegration
{

public:

    using Expression = NeoFOAM::dsl::Expression;

    TimeIntegration(const TimeIntegration& timeIntegrator)
        : timeIntegratorStrategy_(timeIntegrator.timeIntegratorStrategy_->clone()) {};

    TimeIntegration(TimeIntegration&& timeIntegrator)
        : timeIntegratorStrategy_(std::move(timeIntegrator.timeIntegratorStrategy_)) {};

    TimeIntegration(const Dictionary& dict)
        : timeIntegratorStrategy_(
            TimeIntegratorBase<SolutionFieldType>::create(dict.get<std::string>("type"), dict)
        ) {};

    void solve(Expression& eqn, SolutionFieldType& sol, scalar t, scalar dt)
    {
        timeIntegratorStrategy_->solve(eqn, sol, t, dt);
    }

private:

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> timeIntegratorStrategy_;
};


} // namespace NeoFOAM::dsl
