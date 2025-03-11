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
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
    NeoFOAM::fill(fieldA, 2.0);


    auto fieldAStdSpan = fieldA.span();
    auto fieldANDSpan = NeoFOAM::Span(fieldAStdSpan);

    NeoFOAM::parallelFor(
        exec, {0, 4}, KOKKOS_LAMBDA(const size_t i) { fieldANDSpan[i] *= 2.0; }
    );

    // NOTE: this does not work since throwing from a device function is not supported
    // #ifdef NF_DEBUGC
    //     CHECK_THROWS(NeoFOAM::parallelFor(
    //                      exec, {5, 6}, KOKKOS_LAMBDA(const size_t i) { fieldANDSpan[i] *= 2.0; }
    //     ););
    // #endif

    auto fieldAHost = fieldA.copyToHost();
    auto fieldANDSpanHost = NeoFOAM::Span(fieldAHost.span());

    // to some checking if everything is correct
#ifdef NF_DEBUG
    fieldANDSpanHost.abort = false;
    SECTION("detects out of range") { CHECK_THROWS(fieldANDSpanHost[5]); }
#endif

    SECTION("can access elements")
    {
        REQUIRE(fieldANDSpanHost[0] == 4.0);
        REQUIRE(fieldANDSpanHost[1] == 4.0);
        REQUIRE(fieldANDSpanHost[2] == 4.0);
        REQUIRE(fieldANDSpanHost[3] == 4.0);
        REQUIRE(fieldANDSpanHost[4] == 4.0);
    }
};
