// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core.hpp"

TEST_CASE("Primitives")
{
    SECTION("Scalar", "[Traits]")
    {
        auto one = NeoFOAM::one<NeoFOAM::scalar>::value;

        REQUIRE(one == 1.0);

        auto zero = NeoFOAM::zero<NeoFOAM::scalar>::value;

        REQUIRE(zero == 0.0);
    }

    SECTION("LocalIdx", "[Traits]")
    {
        auto one = NeoFOAM::one<NeoFOAM::localIdx>::value;

        REQUIRE(one == 1);

        auto zero = NeoFOAM::zero<NeoFOAM::localIdx>::value;

        REQUIRE(zero == 0);
    }
}
