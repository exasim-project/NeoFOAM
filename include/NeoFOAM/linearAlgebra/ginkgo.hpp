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

gko::config::pnode parse(const Dictionary& dict);

template<typename ValueType>
class GkoSolverBase
{

private:

    std::shared_ptr<const gko::Executor> gkoExec_;
    Dictionary solverDict_;

    virtual std::shared_ptr<gko::LinOp> solverGen(
        std::shared_ptr<const gko::Executor> exec,
        std::shared_ptr<const gko::LinOp> mtx,
        size_t maxIter,
        float relTol
    ) = 0;

protected:

    GkoSolverBase(Executor exec, Dictionary solverDict)
        : gkoExec_(getGkoExecutor(exec)), solverDict_(solverDict)
    {}

public:

    virtual void solve(LinearSystem<ValueType, int>& sys, Field<ValueType>& x)
    {
        size_t nrows = sys.rhs().size();

        auto solver = solverGen(
            gkoExec_,
            detail::createGkoMtx(gkoExec_, sys),
            size_t(solverDict_.get<int>("maxIters")),
            float(solverDict_.get<double>("relTol"))
        );

        auto rhs = detail::createGkoDense(gkoExec_, sys.rhs().data(), nrows);
        auto gkoX = detail::createGkoDense(gkoExec_, x.data(), nrows);
        solver->apply(rhs, gkoX);
    }
};


template<typename ValueType>
class CG : public GkoSolverBase<ValueType>
{

    virtual std::shared_ptr<gko::LinOp> solverGen(
        std::shared_ptr<const gko::Executor> exec,
        std::shared_ptr<const gko::LinOp> mtx,
        size_t maxIter,
        float relTol
    ) override
    {
        auto fact =
            gko::solver::Bicgstab<ValueType>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(maxIter),
                    gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(relTol)
                )
                .on(exec);
        return fact->generate(mtx);
    }

public:

    CG(Executor exec, Dictionary solverDict) : GkoSolverBase<ValueType>(exec, solverDict) {}
};

template<typename ValueType>
class BiCGStab : public GkoSolverBase<ValueType>
{
    virtual std::shared_ptr<gko::LinOp> solverGen(
        std::shared_ptr<const gko::Executor> exec,
        std::shared_ptr<const gko::LinOp> mtx,
        size_t maxIter,
        float relTol
    )
    {
        auto fact =
            gko::solver::Bicgstab<ValueType>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(maxIter),
                    gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(relTol)
                )
                .on(exec);
        return fact->generate(mtx);
    }

public:

    BiCGStab(Executor exec, Dictionary solverDict) : GkoSolverBase<ValueType>(exec, solverDict) {}
};

}

#endif
