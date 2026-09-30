// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/ordering/preparedPermutation.hpp"

namespace NeoFOAM
{
namespace
{
    // Convert the permutation size to NeoN::localIdx and reject
    // values that cannot be represented.
    NeoN::localIdx checkedSize(NeoN::size_t size)
    {
        if (!std::in_range<NeoN::localIdx>(size))
        {
            throw std::length_error("Permutation size exceeds NeoN::localIdx range");
        }
        return static_cast<NeoN::localIdx>(size);
    }

} // namespace

    PreparedPermutation::PreparedPermutation(
        const Permutation& permutation,
        const NeoN::Executor& exec)
        : oldToNew_(
            exec, 
            permutation.oldToNew().data(),
            checkedSize(permutation.size())
        ),
        newToOld_(
            exec,
            permutation.newToOld().data(),
            checkedSize(permutation.size())
        )
    {}

    NeoN::View<const PreparedPermutation::IndexType> PreparedPermutation::oldToNew() const
    {
        return oldToNew_.view();
    }

    NeoN::View<const PreparedPermutation::IndexType> PreparedPermutation::newToOld() const
    {
        return newToOld_.view();
    }

    PreparedPermutation::SizeType PreparedPermutation::size() const noexcept
    {
        return static_cast<PreparedPermutation::SizeType>(oldToNew_.size());
    }

    const NeoN::Executor& PreparedPermutation::exec() const noexcept
    {
        return oldToNew_.exec();
    }

} // namespace NeoFOAM