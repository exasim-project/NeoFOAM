// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "catch2_common.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

TEST_CASE("tokenList")
{
    NeoFOAM::TokenList tokenList;
    REQUIRE(tokenList.empty());

    tokenList.insert(NeoFOAM::label(1));
    tokenList.insert(NeoFOAM::scalar(2.0));
    tokenList.insert(std::string("string"));

    REQUIRE(tokenList.size() == 3);

    SECTION("GetToken")
    {
        REQUIRE(tokenList.size() == 3);
        REQUIRE(tokenList.get<NeoFOAM::label>(0) == 1);
        REQUIRE(tokenList.get<NeoFOAM::scalar>(1) == 2.0);
        REQUIRE(tokenList.get<std::string>(2) == "string");
    }

    SECTION("RemoveToken")
    {
        tokenList.remove(1);
        REQUIRE(tokenList.size() == 2);
        REQUIRE(tokenList.get<NeoFOAM::label>(0) == 1);
        REQUIRE(tokenList.get<std::string>(1) == "string");
    }

    SECTION("check bad any cast")
    {
        REQUIRE_THROWS_AS(tokenList.get<std::string>(0), std::bad_any_cast);
    }

    SECTION("PopToken")
    {
        REQUIRE(tokenList.size() == 3);
        auto firstToken = tokenList.popFront<NeoFOAM::label>();
        auto secondToken = tokenList.popFront<NeoFOAM::scalar>();
        auto thirdToken = tokenList.popFront<std::string>();
        REQUIRE(firstToken == 1);
        REQUIRE(secondToken == 2.0);
        REQUIRE(thirdToken == "string");
        REQUIRE(tokenList.size() == 0);
    }

    SECTION("Next Token")
    {
        tokenList =
            NeoFOAM::TokenList({NeoFOAM::label(1), NeoFOAM::scalar(2.0), std::string("string")});
        REQUIRE(tokenList.size() == 3);
        REQUIRE(tokenList.next<NeoFOAM::label>() == 1);
        REQUIRE(tokenList.next<NeoFOAM::scalar>() == 2.0);
        REQUIRE(tokenList.next<std::string>() == "string");
        REQUIRE_THROWS_AS(tokenList.next<NeoFOAM::label>(), std::out_of_range);
    }

    SECTION("Access out of bound index")
    {
        REQUIRE_THROWS_AS(tokenList.get<NeoFOAM::label>(3), std::out_of_range);
    }
}
