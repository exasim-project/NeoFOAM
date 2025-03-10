// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <limits>
#include <Kokkos_Core.hpp>

#include "NeoFOAM/NeoFOAM.hpp"


TEST_CASE("parallelFor")
{
    NeoFOAM::Executor exec = NeoFOAM::SerialExecutor {};
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
    NeoFOAM::fill(fieldA, 2.0);

    auto fieldAStdSpan = fieldA.span();
    auto fieldANDSpan = NeoFOAM::Span(fieldAStdSpan);

    SECTION("can access elements")
    {
        REQUIRE(fieldANDSpan[0] == 2.0);
        REQUIRE(fieldANDSpan[1] == 2.0);
        REQUIRE(fieldANDSpan[2] == 2.0);
        REQUIRE(fieldANDSpan[3] == 2.0);
        REQUIRE(fieldANDSpan[4] == 2.0);
    }

#ifdef NF_DEBUG
    fieldANDSpan.abort = false;
    SECTION("detects out of range") { CHECK_THROWS(fieldANDSpan[5]); }
#endif

    SECTION("can be used in parallel for")
    {
        NeoFOAM::parallelFor(
            exec, {0, 5}, KOKKOS_LAMBDA(const size_t i) { fieldANDSpan[i] += 2.0; }
        );
        REQUIRE(fieldANDSpan[0] == 4.0);
        REQUIRE(fieldANDSpan[1] == 4.0);
        REQUIRE(fieldANDSpan[2] == 4.0);
        REQUIRE(fieldANDSpan[3] == 4.0);
        REQUIRE(fieldANDSpan[4] == 4.0);
    }
};
