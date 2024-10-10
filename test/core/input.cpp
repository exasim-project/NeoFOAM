// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/core/primitives/label.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"


struct TestInput
{

    static TestInput read(const NeoFOAM::Dictionary& dict)
    {
        TestInput ti;
        ti.label_ = dict.get<NeoFOAM::label>("label");
        ti.scalar_ = dict.get<NeoFOAM::scalar>("scalar");
        ti.string_ = dict.get<std::string>("string");
        return ti;
    }

    static TestInput read(const NeoFOAM::TokenList& tl)
    {
        TestInput ti;
        ti.label_ = tl.get<NeoFOAM::label>(0);
        ti.scalar_ = tl.get<NeoFOAM::scalar>(1);
        ti.string_ = tl.get<std::string>(2);
        return ti;
    }
    NeoFOAM::label label_;
    NeoFOAM::scalar scalar_;
    std::string string_;
};

TEST_CASE("Input")
{
    SECTION("read from TokenList")
    {
        NeoFOAM::TokenList tokenList;
        REQUIRE(tokenList.empty());

        tokenList.insert(NeoFOAM::label(1));
        tokenList.insert(NeoFOAM::scalar(2.0));
        tokenList.insert(std::string("string"));

        REQUIRE(tokenList.size() == 3);

        NeoFOAM::Input input = tokenList;

        TestInput ti = NeoFOAM::read<TestInput>(input);
        REQUIRE(ti.label_ == 1);
        REQUIRE(ti.scalar_ == 2.0);
        REQUIRE(ti.string_ == "string");
    }

    SECTION("read from Dictionary")
    {
        NeoFOAM::Dictionary dict;

        dict.insert("label", NeoFOAM::label(1));
        dict.insert("scalar", NeoFOAM::scalar(2.0));
        dict.insert("string", std::string("string"));

        REQUIRE(dict.keys().size() == 3);

        NeoFOAM::Input input = dict;

        TestInput ti = NeoFOAM::read<TestInput>(input);
        REQUIRE(ti.label_ == 1);
        REQUIRE(ti.scalar_ == 2.0);
        REQUIRE(ti.string_ == "string");
    }
}
