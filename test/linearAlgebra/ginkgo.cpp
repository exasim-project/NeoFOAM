// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include <NeoFOAM/linearAlgebra/ginkgo.hpp>

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"
#include <catch2/matchers/catch_matchers_templated.hpp>

#if NF_WITH_GINKGO

bool operator==(const gko::config::pnode& a, const gko::config::pnode& b)
{
    using tag_t = gko::config::pnode::tag_t;

    if (a.get_tag() != b.get_tag())
    {
        return false;
    }

    if (a.get_tag() == tag_t::array)
    {
        const auto& a_arr = a.get_array();
        const auto& b_arr = b.get_array();
        if (a_arr.size() != b_arr.size())
        {
            return false;
        }
        for (std::size_t i = 0; i < a_arr.size(); ++i)
        {
            if (!(a_arr[i] == b_arr[i]))
            {
                return false;
            }
        }
    }
    if (a.get_tag() == tag_t::boolean)
    {
        return a.get_boolean() == b.get_boolean();
    }
    if (a.get_tag() == tag_t::integer)
    {
        return a.get_integer() == b.get_integer();
    }
    if (a.get_tag() == tag_t::map)
    {
        const auto& a_map = a.get_map();
        const auto& b_map = b.get_map();
        if (a_map.size() != b_map.size())
        {
            return false;
        }
        for (const auto& [key, value] : a_map)
        {
            if (!b_map.contains(key))
            {
                return false;
            }
            if (!(b_map.at(key) == value))
            {
                return false;
            }
        }
    }
    if (a.get_tag() == tag_t::real)
    {
        return a.get_real() == b.get_real();
    }
    if (a.get_tag() == tag_t::string)
    {
        return a.get_string() == b.get_string();
    }

    return true;
}

struct EqualsPnodeMatcher : Catch::Matchers::MatcherGenericBase
{
    EqualsPnodeMatcher(const gko::config::pnode& node) : node_ {node} {}

    bool match(const gko::config::pnode& other) const { return node_ == other; }

    std::string describe() const override { return "Equals: to node"; }

private:

    const gko::config::pnode& node_;
};

TEST_CASE("Dictionary Parsing - Ginkgo")
{
    SECTION("String")
    {
        NeoFOAM::Dictionary dict {{{"key", "value"}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {"value"}}});
        CHECK_THAT(node, EqualsPnodeMatcher(expected));
    }
    SECTION("Int")
    {
        NeoFOAM::Dictionary dict {{{"key", 10}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {10}}});
        CHECK_THAT(node, EqualsPnodeMatcher(expected));
    }
    SECTION("Double")
    {
        NeoFOAM::Dictionary dict {{{"key", 1.0}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0}}});
        CHECK_THAT(node, EqualsPnodeMatcher(expected));
    }
    SECTION("Float")
    {
        NeoFOAM::Dictionary dict {{{"key", 1.0f}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0f}}});
        CHECK_THAT(node, EqualsPnodeMatcher(expected));
    }
    SECTION("Dict")
    {
        NeoFOAM::Dictionary dict;
        dict.insert("key", NeoFOAM::Dictionary {{"key", "value"}});

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected(
            {{"key", gko::config::pnode({{"key", gko::config::pnode {"value"}}})}}
        );
        CHECK_THAT(node, EqualsPnodeMatcher(expected));
    }
}

#endif
