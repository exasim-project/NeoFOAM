// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/dsl/timeIntegration/timeIntegration.hpp"

namespace NeoFOAM::dsl
{

template<typename EquationType, typename SolutionType>
class ForwardEuler :
    public TimeIntegrationFactory<EquationType, SolutionType>::template Register<
        ForwardEuler<EquationType, SolutionType>>
{

public:

    using Base = TimeIntegrationFactory<EquationType, SolutionType>::template Register<
        ForwardEuler<EquationType, SolutionType>>;

    ForwardEuler(const Dictionary& dict) : Base(dict) {}

    static std::string name() { return "forwardEuler"; }

    static std::string doc() { return "first order time integration method"; }

    static std::string schema() { return "none"; }

    virtual void solve(EquationType& eqn, SolutionType& sol) const override
    {
        scalar dt = eqn.getDt();
        Field<scalar> phi(sol.exec(), sol.size(), 0.0);
        Field<scalar> source = eqn.explicitOperation();

        // phi += source*dt;
        // for (auto& op : eqn.temporalOperators())
        // {
        //     op.temporalOperation(phi);
        // }
        sol.internalField() -= source * dt;
        sol.correctBoundaryConditions();

        // check if executor is GPU
        // if (std::holds_alternative<NeoFOAM::GPUExecutor>(eqnSystem_.exec()))
        // {
        //     Kokkos::fence();
        // }
    };

    std::unique_ptr<TimeIntegrationFactory<EquationType, SolutionType>> clone() const override
    {
        return std::make_unique<ForwardEuler>(*this);
    }
};


} // namespace NeoFOAM
