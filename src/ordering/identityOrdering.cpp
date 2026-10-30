// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/ordering/identityOrdering.hpp"

namespace NeoFOAM
{
Permutation IdentityOrdering::compute(std::size_t size) const
{
    return Permutation::identity(size);
}
} // namespace NeoFOAM