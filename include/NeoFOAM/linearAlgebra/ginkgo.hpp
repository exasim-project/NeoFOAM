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

namespace detail
{
template<typename ValueType, typename IndexType>
std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
createGkoMtx(std::shared_ptr<const gko::Executor> exec, LinearSystem<ValueType, IndexType>& sys)
{
    auto& mtx = sys.matrix();
    size_t nrows = sys.rhs().size();

    auto valuesView = gko::array<scalar>::view(exec, mtx.values().size(), mtx.values().data());
    auto colIdxView = gko::array<int>::view(exec, mtx.colIdxs().size(), mtx.colIdxs().data());
    auto rowPtrView = gko::array<int>::view(exec, mtx.rowPtrs().size(), mtx.rowPtrs().data());

    return gko::share(gko::matrix::Csr<scalar, int>::create(
        exec, gko::dim<2> {nrows, nrows}, valuesView, colIdxView, rowPtrView
    ));
}

template<typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
createGkoDense(std::shared_ptr<const gko::Executor> exec, ValueType* ptr, size_t size)
{
    return gko::share(gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2> {size, 1}, gko::array<scalar>::view(exec, size, ptr), 1
    ));
}
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

        auto A = detail::createGkoMtx(gkoExec_, sys);
        auto solver = solverFact->generate(A);

        auto rhs = detail::createGkoDense(gkoExec_, sys.rhs().data(), nrows);
        auto gkoX = detail::createGkoDense(gkoExec_, x.data(), nrows);
        solver->apply(rhs, gkoX);
    }
};

template<typename ValueType>
class BiCGStab
{

    std::shared_ptr<const gko::Executor> gkoExec_;
    Dictionary solverDict_;

public:

    BiCGStab(Executor exec, Dictionary solverDict)
        : gkoExec_(getGkoExecutor(exec)), solverDict_(solverDict)
    {}

    void solve(LinearSystem<ValueType, int>& sys, Field<ValueType>& x)
    {
        size_t nrows = sys.rhs().size();

        auto maxIters = size_t(solverDict_.get<int>("maxIters"));
        auto relTol = solverDict_.get<float>("relTol");
        auto solverFact =
            gko::solver::Bicgstab<ValueType>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(maxIters),
                    gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(relTol)
                )
                .on(gkoExec_);

        auto A = detail::createGkoMtx(gkoExec_, sys);
        auto solver = solverFact->generate(A);

        auto rhs = detail::createGkoDense(gkoExec_, sys.rhs().data(), nrows);
        auto gkoX = detail::createGkoDense(gkoExec_, x.data(), nrows);
        solver->apply(rhs, gkoX);
    }
};

}

#endif
