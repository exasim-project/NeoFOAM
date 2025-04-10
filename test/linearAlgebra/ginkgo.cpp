// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


#if NF_WITH_GINKGO


TEST_CASE("Dictionary Parsing - Ginkgo")
{
    SECTION("String")
    {
        NeoN::Dictionary dict {{{"key", std::string("value")}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {"value"}}});
        CHECK(node == expected);
    }
    SECTION("Const Char *")
    {
        NeoN::Dictionary dict {{{"key", "value"}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {"value"}}});
        CHECK(node == expected);
    }
    SECTION("Int")
    {
        NeoN::Dictionary dict {{{"key", 10}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {10}}});
        CHECK(node == expected);
    }
    SECTION("Double")
    {
        NeoN::Dictionary dict {{{"key", 1.0}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0}}});
        CHECK(node == expected);
    }
    SECTION("Float")
    {
        NeoN::Dictionary dict {{{"key", 1.0f}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0f}}});
        CHECK(node == expected);
    }
    SECTION("Dict")
    {
        NeoN::Dictionary dict;
        dict.insert("key", NeoN::Dictionary {{"key", "value"}});

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected(
            {{"key", gko::config::pnode({{"key", gko::config::pnode {"value"}}})}}
        );
        CHECK(node == expected);
    }
    SECTION("Throws")
    {
        NeoN::Dictionary dict({{"key", std::pair<int*, std::vector<double>> {}}});

        REQUIRE_THROWS_AS(NeoN::la::ginkgo::parse(dict), NeoN::NeoFOAMException);
    }
}

#endif
