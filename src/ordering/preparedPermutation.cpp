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
            permutation.size()
        ),
        newToOld_(
            exec,
            permutation.newToOld().data(),
            permutation.size()
        )
    {}
} // namespace NeoFOAM
