// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

namespace NeoFOAM::dsl
{

/* @class Factory class to create time integration method by a given name
 * using NeoFOAMs runTimeFactory mechanism
 */
template<typename EquationType, typename SolutionType>
class TimeIntegrationFactory :
    public RuntimeSelectionFactory<
        TimeIntegrationFactory<EquationType, SolutionType>,
        Parameters<const Dictionary&>>
{

public:

    static std::string name() { return "timeIntegrationFactory"; }

    TimeIntegrationFactory(const Dictionary& dict) : dict_(dict) {}

    virtual ~TimeIntegrationFactory() {}

    virtual void
    solve(EquationType& eqn, SolutionType& sol) const = 0; // Pure virtual function for solving

    // Pure virtual function for cloning
    virtual std::unique_ptr<TimeIntegrationFactory> clone() const = 0;

protected:

    const Dictionary& dict_;
};

/* @class Factory class to create time integration method by a given name
 * using NeoFOAMs runTimeFactory mechanism
 *
 * @tparam EquationType Injects the type of equation for into the solve method
 * @tparam SolutionType Type of the solution field eg, volumeField or just a plain Field
 */
template<typename EquationType, typename SolutionType>
class TimeIntegration
{

public:

    TimeIntegration(const TimeIntegration& timeIntegrator)
        : timeIntegratorStrategy_(timeIntegrator.timeIntegratorStrategy_->clone()) {};

    TimeIntegration(TimeIntegration&& timeIntegrator)
        : timeIntegratorStrategy_(std::move(timeIntegrator.timeIntegratorStrategy_)) {};

    TimeIntegration(const Dictionary& dict)
        : timeIntegratorStrategy_(TimeIntegrationFactory<EquationType, SolutionType>::create(
            dict.get<std::string>("type"), dict
        )) {};

    void solve(EquationType& eqn, SolutionType& sol) { timeIntegratorStrategy_->solve(eqn, sol); }

private:

    std::unique_ptr<TimeIntegrationFactory<EquationType, SolutionType>> timeIntegratorStrategy_;
};

} // namespace NeoFOAM::dsl
