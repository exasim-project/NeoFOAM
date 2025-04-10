// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

TEST_CASE("Boundaries")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("domainField_" + execName)
    {

        NeoFOAM::DomainField<double> a(exec, 1000, 100, 10);

        NeoFOAM::fill(a.internalField(), 2.0);
        REQUIRE(equal(a.internalField(), 2.0));
    }

    SECTION("boundaryFields_" + execName)
    {

        NeoFOAM::BoundaryFields<double> bCs(exec, 100, 10);

        NeoFOAM::fill(bCs.value(), 2.0);
        REQUIRE(equal(bCs.value(), 2.0));

        NeoFOAM::fill(bCs.refValue(), 2.0);
        REQUIRE(equal(bCs.refValue(), 2.0));

        NeoFOAM::fill(bCs.refGrad(), 2.0);
        REQUIRE(equal(bCs.refGrad(), 2.0));

        NeoFOAM::fill(bCs.valueFraction(), 2.0);
        REQUIRE(equal(bCs.valueFraction(), 2.0));
    }
}
