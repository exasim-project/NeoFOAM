// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <catch2/catch_test_macros.hpp>
#include "NeoFOAM/core/Dictionary.hpp"

TEST_CASE("Dictionary operations", "[dictionary]")
{
    NeoFOAM::Dictionary dict;

    SECTION("Insert and retrieve values")
    {
        dict.insert("key1", 42);
        dict.insert("key2", std::string("Hello"));

        REQUIRE(std::any_cast<int>(dict["key1"]) == 42);
        REQUIRE(dict.get<int>("key1") == 42);
        REQUIRE(std::any_cast<std::string>(dict["key2"]) == "Hello");
        REQUIRE(dict.get<std::string>("key2") == "Hello");
    }

    SECTION("check values")
    {
        dict.insert("key", 42);
        dict["key"] = 43;

        REQUIRE(dict.found("key"));
    }

    SECTION("Modify values")
    {
        dict.insert("key", 42);
        dict["key"] = 43;

        REQUIRE(dict.get<int>("key") == 43);
    }

    SECTION("remove values")
    {
        dict.insert("key", 42);
        dict["key"] = 43;
        dict.remove("key");

        REQUIRE(!dict.found("key"));
    }

    SECTION("Access non-existent key")
    {
        REQUIRE_THROWS_AS(dict["non_existent_key"], std::out_of_range);
        REQUIRE_THROWS_AS(dict.get<int>("non_existent_key"), std::out_of_range);
    }

    SECTION("subDict")
    {
        NeoFOAM::Dictionary subDict;
        subDict.insert("key1", 42);
        subDict.insert("key2", std::string("Hello"));

        dict.insert("subDict", subDict);

        NeoFOAM::Dictionary& sDict = dict.subDict("subDict");
        REQUIRE(sDict.get<int>("key1") == 42);
        REQUIRE(sDict.get<std::string>("key2") == "Hello");

        sDict.get<int>("key1") = 100;

        // check if the value is modified
        NeoFOAM::Dictionary& sDict2 = dict.subDict("subDict");
        REQUIRE(sDict2.get<int>("key1") == 100);
    }
}
