// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/fields/boundaryFields.hpp"
#include "NeoFOAM/fields/domainField.hpp"

TEST_CASE("Boundaries")
{

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

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
