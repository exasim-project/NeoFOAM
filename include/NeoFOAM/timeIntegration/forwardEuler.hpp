// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/timeIntegration/timeIntegration.hpp"

namespace NeoFOAM::dsl
{

template<typename SolutionType>
class ForwardEuler :
    public TimeIntegratorBase<SolutionType>::template Register<ForwardEuler<SolutionType>>
{

public:

    using Base = TimeIntegratorBase<SolutionType>::template Register<ForwardEuler<SolutionType>>;

    ForwardEuler(const Dictionary& dict) : Base(dict) {}

    static std::string name() { return "forwardEuler"; }

    static std::string doc() { return "first order time integration method"; }

    static std::string schema() { return "none"; }

    void solve(Expression& eqn, SolutionType& sol, scalar dt) const override
    {
        auto source = eqn.explicitOperation(sol.size());

        sol.internalField() -= source * dt;
        sol.correctBoundaryConditions();

        // check if executor is GPU
        if (std::holds_alternative<NeoFOAM::GPUExecutor>(eqn.exec()))
        {
            Kokkos::fence();
        }
    };

    std::unique_ptr<TimeIntegratorBase<SolutionType>> clone() const override
    {
        return std::make_unique<ForwardEuler>(*this);
    }
};

// unfortunately needs explicit instantiation
template class ForwardEuler<finiteVolume::cellCentred::VolumeField<scalar>>;


} // namespace NeoFOAM
