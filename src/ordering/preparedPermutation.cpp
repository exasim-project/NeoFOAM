// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoN/NeoN.hpp"
#include "NeoFOAM/ordering/preparedPermutation.hpp"

namespace NeoFOAM
{
    PreparedPermutation::PreparedPermutation(
        const Permutation& permutation,
        const NeoN::Executor& exec)
        : oldToNew_(
            exec, 
            permutation.oldToNew().data(),
            NeoN::toLocalIdx(permutation.size())
        ),
        newToOld_(
            exec,
            permutation.newToOld().data(),
            NeoN::toLocalIdx(permutation.size())
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