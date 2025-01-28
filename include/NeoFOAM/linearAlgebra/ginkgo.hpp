// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#if NF_WITH_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/linearAlgebra/utilities.hpp"


namespace NeoFOAM::la::ginkgo
{

std::shared_ptr<gko::Executor> getGkoExecutor(Executor exec)
{
    return std::visit(
        [](auto concreteExec)
        { return gko::ext::kokkos::create_executor(concreteExec.underlyingExec()); },
        exec
    );
}

template<typename ValueType>
class CG
{

    std::shared_ptr<const gko::Executor> gkoExec_;
    Dictionary solverDict_;

public:

    CG(Executor exec, Dictionary solverDict)
        : gkoExec_(getGkoExecutor(exec)), solverDict_(solverDict)
    {}

    void solve(LinearSystem<ValueType, int>& sys, Field<ValueType>& x)
    {
        auto& mtx = sys.matrix();

        auto valuesView =
            gko::array<scalar>::view(gkoExec_, mtx.values().size(), mtx.values().data());
        auto colIdxView =
            gko::array<int>::view(gkoExec_, mtx.colIdxs().size(), mtx.colIdxs().data());
        auto rowPtrView =
            gko::array<int>::view(gkoExec_, mtx.rowPtrs().size(), mtx.rowPtrs().data());
        size_t nrows = sys.rhs().size();

        auto maxIters = size_t(solverDict_.get<int>("maxIters"));
        auto relTol = solverDict_.get<float>("relTol");
        auto solverFact =
            gko::solver::Cg<ValueType>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(maxIters),
                    gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(relTol)
                )
                .on(gkoExec_);

        auto gkoA = gko::share(gko::matrix::Csr<scalar, int>::create(
            gkoExec_, gko::dim<2> {nrows, nrows}, valuesView, colIdxView, rowPtrView
        ));

        auto solver = solverFact->generate(gkoA);

        auto gkoRhs = gko::matrix::Dense<scalar>::create(
            gkoExec_,
            gko::dim<2> {nrows, 1},
            gko::array<scalar>::view(gkoExec_, nrows, sys.rhs().data()),
            1
        );
        auto gkoX = gko::matrix::Dense<scalar>::create(
            gkoExec_, gko::dim<2> {nrows, 1}, gko::array<scalar>::view(gkoExec_, nrows, x.data()), 1
        );
        solver->apply(gkoRhs, gkoX);
    }
};

}

#endif
