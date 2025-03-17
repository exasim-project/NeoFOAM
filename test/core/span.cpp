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

    NeoFOAM::Field<NeoFOAM::scalar> field(exec, 5);
    NeoFOAM::fill(field, 2.0);


    auto fieldStdSpan = field.span();
    auto fieldNFSpan = NeoFOAM::Span(fieldStdSpan);

    NeoFOAM::parallelFor(
        exec, {0, 5}, KOKKOS_LAMBDA(const size_t i) { fieldNFSpan[i] *= 2.0; }
    );
    REQUIRE(fieldNFSpan.failureIndex == 0);

#ifdef NF_DEBUGC
// TODO: on MSCV this results in a non terminating loop
// so for now we deactivate it on MSVC since it a debugging helper
#ifndef _MSC_VER
    fieldNFSpan.abort = false;
    NeoFOAM::parallelFor(
        exec, {5, 6}, KOKKOS_LAMBDA(const size_t i) { fieldNFSpan[i] *= 2.0; }
    );
    REQUIRE(fieldNFSpan.failureIndex == 5);
#endif
#endif

    auto fieldHost = field.copyToHost();
    auto fieldNFSpanHost = NeoFOAM::Span(fieldHost.span());

#ifdef NF_DEBUG
// TODO: on MSCV this results in a non terminating loop
// so for now we deactivate it on MSVC since it a debugging helper
#ifndef _MSC_VER
    fieldNFSpanHost.abort = false;
    SECTION("detects out of range")
    {
        auto tmp = fieldNFSpanHost[5];
        REQUIRE(fieldNFSpanHost.failureIndex == 5);
    }
#endif
#endif

    // some checking if everything is correct
    SECTION("can access elements")
    {
        REQUIRE(fieldNFSpanHost[0] == 4.0);
        REQUIRE(fieldNFSpanHost[1] == 4.0);
        REQUIRE(fieldNFSpanHost[2] == 4.0);
        REQUIRE(fieldNFSpanHost[3] == 4.0);
        REQUIRE(fieldNFSpanHost[4] == 4.0);
    }
};
