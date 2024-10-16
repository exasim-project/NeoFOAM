// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/dsl/equation.hpp"

namespace NeoFOAM::dsl
{

/* @class Factory class to create time integration method by a given name
 * using NeoFOAMs runTimeFactory mechanism
 */
template<typename SolutionType>
class TimeIntegrationFactory :
    public RuntimeSelectionFactory<
        TimeIntegrationFactory<SolutionType>,
        Parameters<const Dictionary&>>
{

public:

    static std::string name() { return "timeIntegrationFactory"; }

    TimeIntegrationFactory(const Dictionary& dict) : dict_(dict) {}

    virtual ~TimeIntegrationFactory() {}

    virtual void
    solve(Equation& eqn, SolutionType& sol) const = 0; // Pure virtual function for solving

    // Pure virtual function for cloning
    virtual std::unique_ptr<TimeIntegrationFactory> clone() const = 0;

protected:

    const Dictionary& dict_;
};

/* @class Factory class to create time integration method by a given name
 * using NeoFOAMs runTimeFactory mechanism
 *
 * @tparam SolutionType Type of the solution field eg, volumeField or just a plain Field
 */
template<typename SolutionType>
class TimeIntegration
{

public:

    TimeIntegration(const TimeIntegration& timeIntegrator)
        : timeIntegratorStrategy_(timeIntegrator.timeIntegratorStrategy_->clone()) {};

    TimeIntegration(TimeIntegration&& timeIntegrator)
        : timeIntegratorStrategy_(std::move(timeIntegrator.timeIntegratorStrategy_)) {};

    TimeIntegration(const Dictionary& dict)
        : timeIntegratorStrategy_(
            TimeIntegrationFactory<SolutionType>::create(dict.get<std::string>("type"), dict)
        ) {};

    void solve(Equation& eqn, SolutionType& sol) { timeIntegratorStrategy_->integrate(eqn, sol); }

private:

    std::unique_ptr<TimeIntegrationFactory<SolutionType>> timeIntegratorStrategy_;
};

} // namespace NeoFOAM::dsl
