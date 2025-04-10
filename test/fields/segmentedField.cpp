// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

#include <Kokkos_Core.hpp>

TEST_CASE("segmentedField")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Constructor from sizes " + execName)
    {
        NeoN::SegmentedField<NeoN::label, NeoN::localIdx> segField(exec, 10, 5);
        auto [values, segments] = segField.spans();

        REQUIRE(values.size() == 10);
        REQUIRE(segments.size() == 6);

        REQUIRE(segField.numSegments() == 5);
        REQUIRE(segField.size() == 10);
    }

    SECTION("Constructor from field " + execName)
    {
        NeoN::Field<NeoN::label> values(exec, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        NeoN::Field<NeoN::localIdx> segments(exec, {0, 2, 4, 6, 8, 10});

        NeoN::SegmentedField<NeoN::label, NeoN::localIdx> segField(values, segments);

        REQUIRE(segField.values().size() == 10);
        REQUIRE(segField.segments().size() == 6);
        REQUIRE(segField.numSegments() == 5);

        REQUIRE(segField.values().exec() == exec);

        auto hostValues = segField.values().copyToHost();
        auto hostSegments = segField.segments().copyToHost();

        REQUIRE(hostValues.span()[5] == 5);
        REQUIRE(hostSegments.span()[2] == 4);

        SECTION("loop over segments")
        {
            auto [valueSpan, segment] = segField.spans();
            auto segView = segField.view();
            NeoN::Field<NeoN::label> result(exec, 5);

            NeoN::fill(result, 0);
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
            REQUIRE(hostResult.span()[0] == 1 * 3);
            REQUIRE(hostResult.span()[1] == 5 * 3);
            REQUIRE(hostResult.span()[2] == 9 * 3);
            REQUIRE(hostResult.span()[3] == 13 * 3);
            REQUIRE(hostResult.span()[4] == 17 * 3);
        }
    }

    SECTION("Constructor from list with offsets " + execName)
    {
        NeoN::Field<NeoN::localIdx> offsets(exec, {1, 2, 3, 4, 5});
        NeoN::SegmentedField<NeoN::label, NeoN::localIdx> segField(offsets);

        auto hostSegments = segField.segments().copyToHost();
        REQUIRE(hostSegments.span()[0] == 0);
        REQUIRE(hostSegments.span()[1] == 1);
        REQUIRE(hostSegments.span()[2] == 3);
        REQUIRE(hostSegments.span()[3] == 6);
        REQUIRE(hostSegments.span()[4] == 10);
        REQUIRE(hostSegments.span()[5] == 15);

        auto hostOffsets = offsets.copyToHost();
        REQUIRE(hostOffsets.span()[0] == 1);
        REQUIRE(hostOffsets.span()[1] == 2);
        REQUIRE(hostOffsets.span()[2] == 3);
        REQUIRE(hostOffsets.span()[3] == 4);
        REQUIRE(hostOffsets.span()[4] == 5);

        REQUIRE(segField.size() == 15);

        SECTION("update values")
        {
            auto segView = segField.view();
            NeoN::Field<NeoN::label> result(exec, 5);

            NeoN::fill(result, 0);
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
            REQUIRE(hostResult.span()[0] == 0 * 1);
            REQUIRE(hostResult.span()[1] == 1 * 2);
            REQUIRE(hostResult.span()[2] == 2 * 3);
            REQUIRE(hostResult.span()[3] == 3 * 4);
            REQUIRE(hostResult.span()[4] == 4 * 5);
        }
    }
}
