// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <limits>
#include <Kokkos_Core.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"


TEST_CASE("parallelFor")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("parallelFor_" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto spanA = fieldA.span();
        auto spanB = fieldB.span();
        NeoFOAM::fill(fieldB, 1.0);
        NeoFOAM::parallelFor(
            exec, {0, 5}, KOKKOS_LAMBDA(const size_t i) { spanA[i] = spanB[i] + 2.0; }
        );
        auto hostA = fieldA.copyToHost();
        for (auto value : hostA.span())
        {
            REQUIRE(value == 3.0);
        }
    }

    SECTION("parallelFor_Vector" + execName)
    {
        NeoFOAM::Field<NeoFOAM::Vector> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, NeoFOAM::Vector(0.0, 0.0, 0.0));
        NeoFOAM::Field<NeoFOAM::Vector> fieldB(exec, 5);
        auto spanA = fieldA.span();
        auto spanB = fieldB.span();
        NeoFOAM::fill(fieldB, NeoFOAM::Vector(1.0, 1.0, 1.0));
        NeoFOAM::parallelFor(
            exec,
            {0, 5},
            KOKKOS_LAMBDA(const size_t i) { spanA[i] = spanB[i] + NeoFOAM::Vector(2.0, 2.0, 2.0); }
        );
        auto hostA = fieldA.copyToHost();
        for (auto value : hostA.span())
        {
            REQUIRE(value == NeoFOAM::Vector(3.0, 3.0, 3.0));
        }
    }

    SECTION("parallelFor_Field_" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto spanB = fieldB.span();
        NeoFOAM::fill(fieldB, 1.0);
        NeoFOAM::parallelFor(
            fieldA, KOKKOS_LAMBDA(const size_t i) { return spanB[i] + 2.0; }
        );
        auto hostA = fieldA.copyToHost();
        for (auto value : hostA.span())
        {
            REQUIRE(value == 3.0);
        }
    }
};


TEST_CASE("parallelReduce")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("parallelReduce_" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto spanB = fieldB.span();
        NeoFOAM::fill(fieldB, 1.0);
        NeoFOAM::scalar sum = 0.0;
        NeoFOAM::parallelReduce(
            exec, {0, 5}, KOKKOS_LAMBDA(const size_t i, double& lsum) { lsum += spanB[i]; }, sum
        );

        REQUIRE(sum == 5.0);
    }

    SECTION("parallelReduce_MaxValue" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto spanB = fieldB.span();
        NeoFOAM::fill(fieldB, 1.0);
        auto max = std::numeric_limits<NeoFOAM::scalar>::lowest();
        Kokkos::Max<NeoFOAM::scalar> reducer(max);
        NeoFOAM::parallelReduce(
            exec,
            {0, 5},
            KOKKOS_LAMBDA(const size_t i, NeoFOAM::scalar& lmax) {
                if (lmax < spanB[i]) lmax = spanB[i];
            },
            reducer
        );

        REQUIRE(max == 1.0);
    }

    SECTION("parallelReduce_Field_" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto spanB = fieldB.span();
        NeoFOAM::fill(fieldB, 1.0);
        NeoFOAM::scalar sum = 0.0;
        NeoFOAM::parallelReduce(
            fieldA, KOKKOS_LAMBDA(const size_t i, double& lsum) { lsum += spanB[i]; }, sum
        );

        REQUIRE(sum == 5.0);
    }

    SECTION("parallelReduce_Field_MaxValue" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto spanB = fieldB.span();
        NeoFOAM::fill(fieldB, 1.0);
        auto max = std::numeric_limits<NeoFOAM::scalar>::lowest();
        Kokkos::Max<NeoFOAM::scalar> reducer(max);
        NeoFOAM::parallelReduce(
            fieldA,
            KOKKOS_LAMBDA(const size_t i, NeoFOAM::scalar& lmax) {
                if (lmax < spanB[i]) lmax = spanB[i];
            },
            reducer
        );

        REQUIRE(max == 1.0);
    }
};

TEST_CASE("parallelScan")
{
    NeoFOAM::Executor exec = GENERATE(NeoFOAM::Executor(NeoFOAM::SerialExecutor {})
                                      // NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
                                      // NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);


    SECTION("parallelScan_withoutReturn" + execName)
    {
        NeoFOAM::Field<NeoFOAM::localIdx> intervals(exec, {1, 2, 3, 4, 5});
        NeoFOAM::Field<NeoFOAM::localIdx> segments(exec, intervals.size() + 1, 0);
        auto segSpan = segments.span();
        const auto intSpan = intervals.span();

        NeoFOAM::parallelScan(
            exec,
            {1, segSpan.size()},
            KOKKOS_LAMBDA(const std::size_t i, NeoFOAM::localIdx& update, const bool final) {
                update += intSpan[i - 1];
                if (final)
                {
                    segSpan[i] = update;
                }
            }
        );

        auto hostSegments = segments.copyToHost();
        REQUIRE(hostSegments[0] == 0);
        REQUIRE(hostSegments[1] == 1);
        REQUIRE(hostSegments[2] == 3);
        REQUIRE(hostSegments[3] == 6);
        REQUIRE(hostSegments[4] == 10);
        REQUIRE(hostSegments[5] == 15);

        auto hostIntervals = intervals.copyToHost();
        REQUIRE(hostIntervals[0] == 1);
        REQUIRE(hostIntervals[1] == 2);
        REQUIRE(hostIntervals[2] == 3);
        REQUIRE(hostIntervals[3] == 4);
        REQUIRE(hostIntervals[4] == 5);
    }

    SECTION("parallelScan_withReturn" + execName)
    {
        NeoFOAM::Field<NeoFOAM::localIdx> intervals(exec, {1, 2, 3, 4, 5});
        NeoFOAM::Field<NeoFOAM::localIdx> segments(exec, intervals.size() + 1, 0);
        auto segSpan = segments.span();
        const auto intSpan = intervals.span();
        NeoFOAM::localIdx finalValue = 0;

        NeoFOAM::parallelScan(
            exec,
            {1, segSpan.size()},
            KOKKOS_LAMBDA(const std::size_t i, NeoFOAM::localIdx& update, const bool final) {
                update += intSpan[i - 1];
                if (final)
                {
                    segSpan[i] = update;
                }
            },
            finalValue
        );

        REQUIRE(finalValue == 15);

        auto hostSegments = segments.copyToHost();
        REQUIRE(hostSegments[0] == 0);
        REQUIRE(hostSegments[1] == 1);
        REQUIRE(hostSegments[2] == 3);
        REQUIRE(hostSegments[3] == 6);
        REQUIRE(hostSegments[4] == 10);
        REQUIRE(hostSegments[5] == 15);

        auto hostIntervals = intervals.copyToHost();
        REQUIRE(hostIntervals[0] == 1);
        REQUIRE(hostIntervals[1] == 2);
        REQUIRE(hostIntervals[2] == 3);
        REQUIRE(hostIntervals[3] == 4);
        REQUIRE(hostIntervals[4] == 5);
    }
};
