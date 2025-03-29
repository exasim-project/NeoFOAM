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

template<typename T>
gko::array<T> createGkoArray(std::shared_ptr<const gko::Executor> exec, std::span<T> values)
{
    return gko::make_array_view(exec, values.size(), values.data());
}

template<typename T>
gko::detail::const_array_view<T>
createConstGkoArray(std::shared_ptr<const gko::Executor> exec, std::span<const T> values)
{
    return gko::make_const_array_view(exec, values.size(), values.data());
}

template<typename ValueType, typename IndexType>
std::shared_ptr<gko::matrix::Csr<ValueType, int>>
createGkoMtx(std::shared_ptr<const gko::Executor> exec, LinearSystem<ValueType, IndexType>& sys)
{
    size_t nrows = sys.rhs().size();
    // TODO: avoid copying
    // NOTE: since mtx is a converted copy we need to make sure that mtx data is not
    // deallocated before solving
    auto mtx = convert<ValueType, IndexType, ValueType, int>(sys.exec(), sys.view().A);
    auto vals = createGkoArray(exec, mtx.values().span());
    auto col = createGkoArray(exec, mtx.colIdxs().span());
    auto row = createGkoArray(exec, mtx.rowPtrs().span());
    return gko::share(
        gko::matrix::Csr<ValueType, int>::create(exec, gko::dim<2> {nrows, nrows}, vals, col, row)
    );
}

template<typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
createGkoDense(std::shared_ptr<const gko::Executor> exec, ValueType* ptr, size_t size)
{
    return gko::share(gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2> {size, 1}, createGkoArray(exec, std::span {ptr, size}), 1
    ));
}
}

}

#endif
