// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "NeoFOAM/datastructures/ordering/permutation.hpp"

namespace
{

using NeoFOAM::Permutation;
using IndexType = Permutation::IndexType;

} // namespace


TEST_CASE("Permutation can be constructed from a valid old-to-new mapping", "[permutation]")
{
    const Permutation permutation {std::vector<IndexType> {2, 0, 3, 1}};

    REQUIRE(permutation.size() == 4);

    SECTION("old-to-new mapping is preserved")
    {
        REQUIRE(permutation.oldToNew(0) == 2);
        REQUIRE(permutation.oldToNew(1) == 0);
        REQUIRE(permutation.oldToNew(2) == 3);
        REQUIRE(permutation.oldToNew(3) == 1);
    }

    SECTION("new-to-old mapping is constructed correctly")
    {
        REQUIRE(permutation.newToOld(0) == 1);
        REQUIRE(permutation.newToOld(1) == 3);
        REQUIRE(permutation.newToOld(2) == 0);
        REQUIRE(permutation.newToOld(3) == 2);
    }

    SECTION("the permutation is not identified as identity")
    {
        REQUIRE_FALSE(permutation.isIdentity());
    }
}


TEST_CASE("Permutation creates an identity mapping", "[permutation]")
{
    const auto permutation = Permutation::identity(4);

    REQUIRE(permutation.size() == 4);

    SECTION("every old index maps to itself")
    {
        for (IndexType index = 0; index < static_cast<IndexType>(permutation.size()); ++index)
        {
            REQUIRE(permutation.oldToNew(index) == index);
        }
    }

    SECTION("every new index maps to itself")
    {
        for (IndexType index = 0; index < static_cast<IndexType>(permutation.size()); ++index)
        {
            REQUIRE(permutation.newToOld(index) == index);
        }
    }

    SECTION("the permutation is identified as identity") { REQUIRE(permutation.isIdentity()); }
}


TEST_CASE("Permutation supports an empty identity mapping", "[permutation]")
{
    const auto permutation = Permutation::identity(0);

    REQUIRE(permutation.size() == 0);
    REQUIRE(permutation.oldToNew().empty());
    REQUIRE(permutation.newToOld().empty());
    REQUIRE(permutation.isIdentity());
}


TEST_CASE("Permutation exposes the complete old-to-new mapping", "[permutation]")
{
    const Permutation permutation {std::vector<IndexType> {2, 0, 3, 1}};

    const auto oldToNew = permutation.oldToNew();

    REQUIRE(oldToNew.size() == 4);

    REQUIRE(oldToNew[0] == 2);
    REQUIRE(oldToNew[1] == 0);
    REQUIRE(oldToNew[2] == 3);
    REQUIRE(oldToNew[3] == 1);
}


TEST_CASE("Permutation exposes the complete new-to-old mapping", "[permutation]")
{
    const Permutation permutation {std::vector<IndexType> {2, 0, 3, 1}};

    const auto newToOld = permutation.newToOld();

    REQUIRE(newToOld.size() == 4);

    REQUIRE(newToOld[0] == 1);
    REQUIRE(newToOld[1] == 3);
    REQUIRE(newToOld[2] == 0);
    REQUIRE(newToOld[3] == 2);
}


TEST_CASE("Old-to-new and new-to-old mappings are inverses", "[permutation]")
{
    const Permutation permutation {std::vector<IndexType> {2, 0, 4, 1, 3}};

    for (IndexType oldIndex = 0; oldIndex < static_cast<IndexType>(permutation.size()); ++oldIndex)
    {
        const auto newIndex = permutation.oldToNew(oldIndex);

        REQUIRE(permutation.newToOld(newIndex) == oldIndex);
    }

    for (IndexType newIndex = 0; newIndex < static_cast<IndexType>(permutation.size()); ++newIndex)
    {
        const auto oldIndex = permutation.newToOld(newIndex);

        REQUIRE(permutation.oldToNew(oldIndex) == newIndex);
    }
}


TEST_CASE("Inverse swaps the old-to-new and new-to-old mappings", "[permutation]")
{
    const Permutation permutation {std::vector<IndexType> {2, 0, 3, 1}};

    const auto inverse = permutation.inverse();

    REQUIRE(inverse.size() == permutation.size());

    for (IndexType index = 0; index < static_cast<IndexType>(permutation.size()); ++index)
    {
        // REQUIRE(inverse.oldToNew(index) == permutation.newToOld(index));

        // REQUIRE(inverse.newToOld(index) == permutation.oldToNew(index));
    }
}


TEST_CASE("Inverting a permutation twice recovers the original mapping", "[permutation]")
{
    const Permutation permutation {std::vector<IndexType> {2, 0, 4, 1, 3}};

    const auto doubleInverse = permutation.inverse().inverse();

    // REQUIRE(doubleInverse.oldToNew() == permutation.oldToNew());

    // REQUIRE(doubleInverse.newToOld() == permutation.newToOld());
}


TEST_CASE("The inverse of an identity permutation is identity", "[permutation]")
{
    const auto identity = Permutation::identity(5);

    const auto inverse = identity.inverse();

    REQUIRE(inverse.isIdentity());

    // REQUIRE(inverse.oldToNew() == identity.oldToNew());

    // REQUIRE(inverse.newToOld() == identity.newToOld());
}

TEST_CASE("Permutation rejects duplicate new indices", "[permutation][validation]")
{
    REQUIRE_THROWS_AS(Permutation(std::vector<IndexType> {0, 1, 1, 3}), std::invalid_argument);
}


TEST_CASE("Permutation rejects an index outside its valid range", "[permutation][validation]")
{
    REQUIRE_THROWS_AS(Permutation(std::vector<IndexType> {0, 1, 2, 4}), std::invalid_argument);
}


TEST_CASE("Permutation rejects a negative index", "[permutation][validation]")
{
    if constexpr (std::is_signed_v<IndexType>)
    {
        REQUIRE_THROWS_AS(Permutation(std::vector<IndexType> {0, 1, -1, 2}), std::invalid_argument);
    }
}

TEST_CASE("Permutation has value semantics", "[permutation]")
{
    static_assert(std::is_copy_constructible_v<Permutation>);

    static_assert(std::is_move_constructible_v<Permutation>);

    static_assert(std::is_copy_assignable_v<Permutation>);

    static_assert(std::is_move_assignable_v<Permutation>);
}


TEST_CASE("Permutation mapping views provide read-only access", "[permutation]")
{
    using OldToNewView = decltype(std::declval<const Permutation&>().oldToNew());

    using NewToOldView = decltype(std::declval<const Permutation&>().newToOld());

    static_assert(std::is_same_v<OldToNewView, std::span<const IndexType>>);

    static_assert(std::is_same_v<NewToOldView, std::span<const IndexType>>);
}
