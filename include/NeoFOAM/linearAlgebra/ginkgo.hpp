// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#if NF_WITH_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"


namespace NeoN::la::ginkgo
{

gko::config::pnode parse(const Dictionary& dict);

template<typename ValueType>
class Solver
{

public:

    Solver(Executor exec, Dictionary solverConfig)
        : gkoExec_(getGkoExecutor(exec)), config_(parse(solverConfig)),
          factory_(
              gko::config::parse(
                  config_, gko::config::registry(), gko::config::make_type_descriptor<ValueType>()
              )
                  .on(gkoExec_)
          )
    {}

    void solve(LinearSystem<ValueType, localIdx>& sys, Field<ValueType>& x)
    {
        size_t nrows = sys.rhs().size();

        auto gkoMtx = detail::createGkoMtx(gkoExec_, sys);
        auto solver = factory_->generate(gkoMtx);

        auto rhs = detail::createGkoDense(gkoExec_, sys.rhs().data(), nrows);
        auto gkoX = detail::createGkoDense(gkoExec_, x.data(), nrows);

        solver->apply(rhs, gkoX);
    }

private:

    std::shared_ptr<const gko::Executor> gkoExec_;
    gko::config::pnode config_;
    std::shared_ptr<const gko::LinOpFactory> factory_;
};


}

#endif
