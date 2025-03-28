// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
#include "catch2_common.hpp"

#include "NeoFOAM/NeoFOAM.hpp"


#if NF_WITH_GINKGO


TEST_CASE("Dictionary Parsing - Ginkgo")
{
    SECTION("String")
    {
        NeoFOAM::Dictionary dict {{{"key", std::string("value")}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {"value"}}});
        CHECK(node == expected);
    }
    SECTION("Const Char *")
    {
        NeoFOAM::Dictionary dict {{{"key", "value"}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {"value"}}});
        CHECK(node == expected);
    }
    SECTION("Int")
    {
        NeoFOAM::Dictionary dict {{{"key", 10}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {10}}});
        CHECK(node == expected);
    }
    SECTION("Double")
    {
        NeoFOAM::Dictionary dict {{{"key", 1.0}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0}}});
        CHECK(node == expected);
    }
    SECTION("Float")
    {
        NeoFOAM::Dictionary dict {{{"key", 1.0f}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0f}}});
        CHECK(node == expected);
    }
    SECTION("Dict")
    {
        NeoFOAM::Dictionary dict;
        dict.insert("key", NeoFOAM::Dictionary {{"key", "value"}});

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected(
            {{"key", gko::config::pnode({{"key", gko::config::pnode {"value"}}})}}
        );
        CHECK(node == expected);
    }
    SECTION("Throws")
    {
        NeoFOAM::Dictionary dict({{"key", std::pair<int*, std::vector<double>> {}}});

        REQUIRE_THROWS_AS(NeoFOAM::la::ginkgo::parse(dict), NeoFOAM::NeoFOAMException);
    }
}

#endif
