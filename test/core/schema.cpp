// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoFOAM/core/schema/Number.hpp"


TEST_CASE("Number")
{
    NeoFOAM::Number number;
    number.set(NeoFOAM::Description {"description"});
    number.set(NeoFOAM::MultipleOf {1.0});
    number.set(NeoFOAM::Minimum {1.0});
    number.set(NeoFOAM::Maximum {1.0});
    number.set(NeoFOAM::ExclusiveMinimum {1.0});
    number.set(NeoFOAM::ExclusiveMaximum {1.0});

    NeoFOAM::Number number2 = NeoFOAM::Number(
        NeoFOAM::Description {"description"},
        NeoFOAM::MultipleOf {1.0},
        NeoFOAM::Minimum {1.0},
        NeoFOAM::Maximum {1.0},
        NeoFOAM::ExclusiveMinimum {1.0},
        NeoFOAM::ExclusiveMaximum {1.0}
    );

    nlohmann::json j;
    j["test"] = number;
    j["test2"] = number2;

    REQUIRE(j["test"]["type"] == "number");
    REQUIRE(j["test"]["description"] == "description");
    REQUIRE(j["test"]["multipleOf"] == 1.0);
    REQUIRE(j["test"]["minimum"] == 1.0);
    REQUIRE(j["test"]["maximum"] == 1.0);
    REQUIRE(j["test"]["exclusiveMinimum"] == 1.0);
    REQUIRE(j["test"]["exclusiveMaximum"] == 1.0);

    REQUIRE(j["test2"]["type"] == "number");
    REQUIRE(j["test2"]["description"] == "description");
    REQUIRE(j["test2"]["multipleOf"] == 1.0);
    REQUIRE(j["test2"]["minimum"] == 1.0);
    REQUIRE(j["test2"]["maximum"] == 1.0);
    REQUIRE(j["test2"]["exclusiveMinimum"] == 1.0);
    REQUIRE(j["test2"]["exclusiveMaximum"] == 1.0);
}
