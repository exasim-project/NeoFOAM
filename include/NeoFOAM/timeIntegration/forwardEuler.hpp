// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/database/fieldCollection.hpp"
#include "NeoFOAM/core/database/oldTimeCollection.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/timeIntegration/timeIntegration.hpp"

namespace NeoFOAM::timeIntegration
{

template<typename SolutionFieldType>
class ForwardEuler :
    public TimeIntegratorBase<SolutionFieldType>::template Register<ForwardEuler<SolutionFieldType>>
{

public:

    using Expression = NeoFOAM::dsl::Expression;
    using Base =
        TimeIntegratorBase<SolutionFieldType>::template Register<ForwardEuler<SolutionFieldType>>;

    ForwardEuler(const Dictionary& dict) : Base(dict) {}

    static std::string name() { return "forwardEuler"; }

    static std::string doc() { return "first order time integration method"; }

    static std::string schema() { return "none"; }

    void solve(
        Expression& eqn, SolutionFieldType& solutionField, [[maybe_unused]] scalar t, scalar dt
    ) override
    {
        auto source = eqn.explicitOperation(solutionField.size());
        SolutionFieldType& oldSolutionField =
            NeoFOAM::finiteVolume::cellCentred::oldTime(solutionField);

        solutionField.internalField() = oldSolutionField.internalField() - source * dt;
        solutionField.correctBoundaryConditions();

        // check if executor is GPU
        if (std::holds_alternative<NeoFOAM::GPUExecutor>(eqn.exec()))
        {
            Kokkos::fence();
        }
        oldSolutionField.internalField() = solutionField.internalField();
    };

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> clone() const override
    {
        return std::make_unique<ForwardEuler>(*this);
    }
};


} // namespace NeoFOAM
