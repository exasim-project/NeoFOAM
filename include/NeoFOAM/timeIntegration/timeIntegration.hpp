// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/dsl/expression.hpp"

namespace NeoFOAM::dsl
{

/* @class Factory class to create time integration method by a given name
 * using NeoFOAMs runTimeFactory mechanism
 */
template<typename SolutionType>
class TimeIntegratorBase :
    public RuntimeSelectionFactory<TimeIntegratorBase<SolutionType>, Parameters<const Dictionary&>>
{

public:

    static std::string name() { return "timeIntegrationFactory"; }

    TimeIntegratorBase(const Dictionary& dict) : dict_(dict) {}

    virtual ~TimeIntegratorBase() {}

    virtual void
    solve(Expression& eqn, SolutionType& sol, scalar dt) = 0; // Pure virtual function for solving

    // Pure virtual function for cloning
    virtual std::unique_ptr<TimeIntegratorBase> clone() const = 0;

protected:

    const Dictionary& dict_;
};

/* @class Factory class to create time integration method by a given name
 * using NeoFOAMs runTimeFactory mechanism
 *
 * @tparam SolutionFieldType Type of the solution field eg, volumeField or just a plain Field
 */
template<typename SolutionFieldType>
class TimeIntegration
{

public:

    TimeIntegration(const TimeIntegration& timeIntegrator)
        : timeIntegratorStrategy_(timeIntegrator.timeIntegratorStrategy_->clone()) {};

    TimeIntegration(TimeIntegration&& timeIntegrator)
        : timeIntegratorStrategy_(std::move(timeIntegrator.timeIntegratorStrategy_)) {};

    TimeIntegration(const Dictionary& dict)
        : timeIntegratorStrategy_(
            TimeIntegratorBase<SolutionFieldType>::create(dict.get<std::string>("type"), dict)
        ) {};

    void solve(Expression& eqn, SolutionFieldType& sol, scalar dt)
    {
        timeIntegratorStrategy_->solve(eqn, sol, dt);
    }

private:

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> timeIntegratorStrategy_;
};


} // namespace NeoFOAM::dsl
