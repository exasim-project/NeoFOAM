// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoFOAM/ordering/permutation.hpp"
#include <cstddef>

namespace NeoFOAM
{
class IdentityOrdering
{

public:

    [[nodiscard]] Permutation compute(std::size_t size) const;
};
} // namespace NeoFOAM