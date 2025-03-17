// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/database/fieldCollection.hpp"
#include "NeoFOAM/core/database/oldTimeCollection.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/timeIntegration/timeIntegration.hpp"
#include "NeoFOAM/dsl/solver.hpp"

#if NF_WITH_GINKGO
#include "NeoFOAM/linearAlgebra/ginkgo.hpp"
#endif

namespace NeoFOAM::timeIntegration
{

template<typename SolutionFieldType>
class BackwardEuler :
    public TimeIntegratorBase<SolutionFieldType>::template Register<
        BackwardEuler<SolutionFieldType>>
{

public:

    using ValueType = typename SolutionFieldType::FieldValueType;
    using Base =
        TimeIntegratorBase<SolutionFieldType>::template Register<BackwardEuler<SolutionFieldType>>;

    BackwardEuler(const Dictionary& schemeDict, const Dictionary& solutionDict)
        : Base(schemeDict, solutionDict)
    {}

    static std::string name() { return "backwardEuler"; }

    static std::string doc() { return "first order time integration method"; }

    static std::string schema() { return "none"; }

    void solve(
        dsl::Expression<ValueType>& eqn,
        SolutionFieldType& solutionField,
        [[maybe_unused]] scalar t,
        scalar dt
    ) override
    {
        auto source = eqn.explicitOperation(solutionField.size());
        SolutionFieldType& oldSolutionField =
            NeoFOAM::finiteVolume::cellCentred::oldTime(solutionField);

        // solutionField.internalField() = oldSolutionField.internalField() - source * dt;
        // solutionField.correctBoundaryConditions();
        // solve sparse matrix system
        using ValueType = typename SolutionFieldType::ElementType;
        auto ls = eqn.implicitOperation();
        auto values = ls.matrix().values();
        eqn.implicitOperation(ls, t, dt);
        auto ginkgoLs = NeoFOAM::dsl::ginkgoMatrix(ls, solutionField);


        NeoFOAM::la::ginkgo::Solver<ValueType> solver(solutionField.exec(), this->solutionDict_);
        solver.solve(ginkgoLs, solutionField.internalField());

        // check if executor is GPU
        if (std::holds_alternative<NeoFOAM::GPUExecutor>(eqn.exec()))
        {
            Kokkos::fence();
        }
        oldSolutionField.internalField() = solutionField.internalField();
    };

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> clone() const override
    {
        return std::make_unique<BackwardEuler>(*this);
    }
};


} // namespace NeoFOAM
