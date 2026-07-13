// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER
#include "common.hpp"

#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

#include "NeoFOAM/datastructures/ordering/permutation.hpp"

namespace {

template<typename T, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<decltype(std::declval<const T&>().size())>> : std::true_type {};

template<typename T, typename = void>
struct has_index_operator : std::false_type {};

template<typename T>
struct has_index_operator<T, std::void_t<decltype(std::declval<const T&>()[std::declval<label>()])>>
    : std::true_type {};

template<typename T, typename = void>
struct has_is_identity : std::false_type {};

template<typename T>
struct has_is_identity<T, std::void_t<decltype(std::declval<const T&>().isIdentity())>>
    : std::true_type {};

template<typename T>
auto sizeOf(const T& value) -> decltype(value.size())
{
    return value.size();
}

template<typename T>
auto indexOf(const T& value, label index) -> decltype(value[index])
{
    return value[index];
}

template<typename T>
auto makeIdentityPermutation(label size)
    -> typename std::enable_if<std::is_constructible<T, label>::value, T>::type
{
    return T(size);
}

template<typename T>
auto makeIdentityPermutation(label)
    -> typename std::enable_if<!std::is_constructible<T, label>::value && std::is_default_constructible<T>::value, T>::type
{
    return T{};
}

template<typename T>
auto makePermutationFromValues(std::initializer_list<label> values)
    -> typename std::enable_if<std::is_constructible<T, std::initializer_list<label>>::value, T>::type
{
    return T{values};
}

template<typename T>
auto makePermutationFromValues(std::initializer_list<label> values)
    -> typename std::enable_if<!std::is_constructible<T, std::initializer_list<label>>::value &&
                                std::is_constructible<T, std::vector<label>>::value, T>::type
{
    return T(std::vector<label>(values));
}

template<typename T>
auto makePermutationFromValues(std::initializer_list<label>)
    -> typename std::enable_if<!std::is_constructible<T, std::initializer_list<label>>::value &&
                                !std::is_constructible<T, std::vector<label>>::value, T>::type
{
    return T{};
}

} // namespace

TEST_CASE("Permutation exposes identity semantics", "[permutation]")
{
    using Permutation = NeoFOAM::Permutation;

    auto permutation = makeIdentityPermutation<Permutation>(4);

    if constexpr (has_size<Permutation>::value)
    {
        REQUIRE(sizeOf(permutation) == 4);
    }

    if constexpr (has_index_operator<Permutation>::value)
    {
        REQUIRE(indexOf(permutation, 0) == 0);
        REQUIRE(indexOf(permutation, 1) == 1);
        REQUIRE(indexOf(permutation, 2) == 2);
        REQUIRE(indexOf(permutation, 3) == 3);
    }

    if constexpr (has_is_identity<Permutation>::value)
    {
        REQUIRE(permutation.isIdentity());
    }
}

TEST_CASE("Permutation preserves an explicit ordering", "[permutation]")
{
    using Permutation = NeoFOAM::Permutation;

    auto permutation = makePermutationFromValues<Permutation>({2, 0, 3, 1});

    if constexpr (has_size<Permutation>::value)
    {
        REQUIRE(sizeOf(permutation) == 4);
    }

    if constexpr (has_index_operator<Permutation>::value)
    {
        REQUIRE(indexOf(permutation, 0) == 2);
        REQUIRE(indexOf(permutation, 1) == 0);
        REQUIRE(indexOf(permutation, 2) == 3);
        REQUIRE(indexOf(permutation, 3) == 1);
    }

    if constexpr (has_is_identity<Permutation>::value)
    {
        REQUIRE_FALSE(permutation.isIdentity());
    }
}

