// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/fields/segmentedField.hpp"
#include "NeoFOAM/core/primitives/label.hpp"
#include <Kokkos_Core.hpp>

TEST_CASE("segmentedField")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("Constructor from sizes " + execName)
    {
        NeoFOAM::SegmentedField<NeoFOAM::label, NeoFOAM::localIdx> segField(exec, 10, 5);
        auto [values, segments] = segField.spans();

        REQUIRE(values.size() == 10);
        REQUIRE(segments.size() == 6);

        REQUIRE(segField.numSegments() == 5);
        REQUIRE(segField.size() == 10);
    }

    SECTION("Constructor from field " + execName)
    {
        NeoFOAM::Field<NeoFOAM::label> values(exec, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        NeoFOAM::Field<NeoFOAM::localIdx> segments(exec, {0, 2, 4, 6, 8, 10});

        NeoFOAM::SegmentedField<NeoFOAM::label, NeoFOAM::localIdx> segField(values, segments);

        REQUIRE(segField.values().size() == 10);
        REQUIRE(segField.segments().size() == 6);
        REQUIRE(segField.numSegments() == 5);

        REQUIRE(segField.values().exec() == exec);

        auto hostValues = segField.values().copyToHost();
        auto hostSegments = segField.segments().copyToHost();

        REQUIRE(hostValues[5] == 5);
        REQUIRE(hostSegments[2] == 4);

        SECTION("loop over segments")
        {
            auto [valueSpan, segment] = segField.spans();
            auto segView = segField.view();
            NeoFOAM::Field<NeoFOAM::label> result(exec, 5);

            NeoFOAM::fill(result, 0);
            auto resultSpan = result.span();

            parallelFor(
                exec,
                {0, segField.numSegments()},
                KOKKOS_LAMBDA(const size_t segI) {
                    // check if it works with bounds
                    auto [bStart, bEnd] = segView.bounds(segI);
                    auto bVals = valueSpan.subspan(bStart, bEnd - bStart);
                    for (auto& val : bVals)
                    {
                        resultSpan[segI] += val;
                    }

                    // check if it works with range
                    auto [rStart, rLength] = segView.range(segI);
                    auto rVals = valueSpan.subspan(rStart, rLength);
                    for (auto& val : rVals)
                    {
                        resultSpan[segI] += val;
                    }

                    // check with subspan
                    auto vals = segView.span(segI);
                    for (auto& val : vals)
                    {
                        resultSpan[segI] += val;
                    }
                }
            );

            auto hostResult = result.copyToHost();
            REQUIRE(hostResult[0] == 1 * 3);
            REQUIRE(hostResult[1] == 5 * 3);
            REQUIRE(hostResult[2] == 9 * 3);
            REQUIRE(hostResult[3] == 13 * 3);
            REQUIRE(hostResult[4] == 17 * 3);
        }
    }

    SECTION("Constructor from list with offsets " + execName)
    {
        NeoFOAM::Field<NeoFOAM::localIdx> offsets(exec, {1, 2, 3, 4, 5});
        NeoFOAM::SegmentedField<NeoFOAM::label, NeoFOAM::localIdx> segField(offsets);

        auto hostSegments = segField.segments().copyToHost();
        REQUIRE(hostSegments[0] == 0);
        REQUIRE(hostSegments[1] == 1);
        REQUIRE(hostSegments[2] == 3);
        REQUIRE(hostSegments[3] == 6);
        REQUIRE(hostSegments[4] == 10);
        REQUIRE(hostSegments[5] == 15);

        auto hostOffsets = offsets.copyToHost();
        REQUIRE(hostOffsets[0] == 1);
        REQUIRE(hostOffsets[1] == 2);
        REQUIRE(hostOffsets[2] == 3);
        REQUIRE(hostOffsets[3] == 4);
        REQUIRE(hostOffsets[4] == 5);

        REQUIRE(segField.size() == 15);

        SECTION("update values")
        {
            auto segView = segField.view();
            NeoFOAM::Field<NeoFOAM::label> result(exec, 5);

            NeoFOAM::fill(result, 0);
            auto resultSpan = result.span();


            parallelFor(
                exec,
                {0, segField.numSegments()},
                KOKKOS_LAMBDA(const size_t segI) {
                    // fill values
                    auto vals = segView.span(segI);
                    for (auto& val : vals)
                    {
                        val = segI;
                    }

                    // accumulate values
                    for (const auto& val : vals)
                    {
                        resultSpan[segI] += val;
                    }
                }
            );

            auto hostResult = result.copyToHost();
            REQUIRE(hostResult[0] == 0 * 1);
            REQUIRE(hostResult[1] == 1 * 2);
            REQUIRE(hostResult[2] == 2 * 3);
            REQUIRE(hostResult[3] == 3 * 4);
            REQUIRE(hostResult[4] == 4 * 5);
        }
    }
}
