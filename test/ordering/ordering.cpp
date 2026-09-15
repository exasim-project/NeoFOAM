// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include <cstddef>

#include "NeoFOAM/ordering/identityOrdering.hpp"
#include "NeoFOAM/ordering/permutation.hpp"

namespace
{
using NeoFOAM::IdentityOrdering;
using NeoFOAM::Permutation;
using IndexType = Permutation::IndexType;
} // namespace

TEST_CASE("Identity ordering creates an identity permutation", "[ordering]")
{
    const IdentityOrdering ordering {};

    SECTION("Empty ordering")
    {
        const auto permutation = ordering.compute(0);

        REQUIRE(permutation.size() == 0);
    }

    SECTION("Single-element ordering")
    {
        const auto permutation = ordering.compute(1);

        REQUIRE(permutation.size() == 1);

        REQUIRE(permutation.oldToNew(0) == 0);
        REQUIRE(permutation.newToOld(0) == 0);
    }

    SECTION("Multi-element ordering")
    {
        constexpr std::size_t size = 5;

        const auto permutation = ordering.compute(size);

        REQUIRE(permutation.size() == size);

        for (IndexType index = 0; index < permutation.size(); ++index)
        {
            REQUIRE(permutation.oldToNew(index) == index);
            REQUIRE(permutation.newToOld(index) == index);
        }
    }
}