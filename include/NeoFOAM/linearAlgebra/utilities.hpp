// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 NeoFOAM authors

#pragma once

#if NF_WITH_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"


namespace NeoFOAM::la
{

std::shared_ptr<gko::Executor> getGkoExecutor(Executor exec);

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

}

#endif
