// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include <Kokkos_Core.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

#include <limits>

TEST_CASE("parallelFor")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("parallelFor_" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto viewA = fieldA.view();
        auto viewB = fieldB.view();
        NeoFOAM::fill(fieldB, 1.0);
        NeoFOAM::parallelFor(
            exec, {0, 5}, KOKKOS_LAMBDA(const size_t i) { viewA[i] = viewB[i] + 2.0; }
        );
        auto hostA = fieldA.copyToHost();
        for (auto value : hostA.view())
        {
            REQUIRE(value == 3.0);
        }
    }

    SECTION("parallelFor_Vector" + execName)
    {
        NeoFOAM::Field<NeoFOAM::Vector> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, NeoFOAM::Vector(0.0, 0.0, 0.0));
        NeoFOAM::Field<NeoFOAM::Vector> fieldB(exec, 5);
        auto viewA = fieldA.view();
        auto viewB = fieldB.view();
        NeoFOAM::fill(fieldB, NeoFOAM::Vector(1.0, 1.0, 1.0));
        NeoFOAM::parallelFor(
            exec,
            {0, 5},
            KOKKOS_LAMBDA(const size_t i) { viewA[i] = viewB[i] + NeoFOAM::Vector(2.0, 2.0, 2.0); }
        );
        auto hostA = fieldA.copyToHost();
        for (auto value : hostA.view())
        {
            REQUIRE(value == NeoFOAM::Vector(3.0, 3.0, 3.0));
        }
    }

    SECTION("parallelFor_Field_" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoFOAM::fill(fieldB, 1.0);
        NeoFOAM::parallelFor(
            fieldA, KOKKOS_LAMBDA(const size_t i) { return viewB[i] + 2.0; }
        );
        auto hostA = fieldA.copyToHost();
        for (auto value : hostA.view())
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
        auto viewB = fieldB.view();
        NeoFOAM::fill(fieldB, 1.0);
        NeoFOAM::scalar sum = 0.0;
        NeoFOAM::parallelReduce(
            exec, {0, 5}, KOKKOS_LAMBDA(const size_t i, double& lsum) { lsum += viewB[i]; }, sum
        );

        REQUIRE(sum == 5.0);
    }

    SECTION("parallelReduce_MaxValue" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoFOAM::fill(fieldB, 1.0);
        auto max = std::numeric_limits<NeoFOAM::scalar>::lowest();
        Kokkos::Max<NeoFOAM::scalar> reducer(max);
        NeoFOAM::parallelReduce(
            exec,
            {0, 5},
            KOKKOS_LAMBDA(const size_t i, NeoFOAM::scalar& lmax) {
                if (lmax < viewB[i]) lmax = viewB[i];
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
        auto viewB = fieldB.view();
        NeoFOAM::fill(fieldB, 1.0);
        NeoFOAM::scalar sum = 0.0;
        NeoFOAM::parallelReduce(
            fieldA, KOKKOS_LAMBDA(const size_t i, double& lsum) { lsum += viewB[i]; }, sum
        );

        REQUIRE(sum == 5.0);
    }

    SECTION("parallelReduce_Field_MaxValue" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoFOAM::fill(fieldB, 1.0);
        auto max = std::numeric_limits<NeoFOAM::scalar>::lowest();
        Kokkos::Max<NeoFOAM::scalar> reducer(max);
        NeoFOAM::parallelReduce(
            fieldA,
            KOKKOS_LAMBDA(const size_t i, NeoFOAM::scalar& lmax) {
                if (lmax < viewB[i]) lmax = viewB[i];
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
        auto segView = segments.view();
        const auto intView = intervals.view();

        NeoFOAM::parallelScan(
            exec,
            {1, segView.size()},
            KOKKOS_LAMBDA(const std::size_t i, NeoFOAM::localIdx& update, const bool final) {
                update += intView[i - 1];
                if (final)
                {
                    segView[i] = update;
                }
            }
        );

        auto hostSegments = segments.copyToHost();
        REQUIRE(hostSegments.view()[0] == 0);
        REQUIRE(hostSegments.view()[1] == 1);
        REQUIRE(hostSegments.view()[2] == 3);
        REQUIRE(hostSegments.view()[3] == 6);
        REQUIRE(hostSegments.view()[4] == 10);
        REQUIRE(hostSegments.view()[5] == 15);

        auto hostIntervals = intervals.copyToHost();
        REQUIRE(hostIntervals.view()[0] == 1);
        REQUIRE(hostIntervals.view()[1] == 2);
        REQUIRE(hostIntervals.view()[2] == 3);
        REQUIRE(hostIntervals.view()[3] == 4);
        REQUIRE(hostIntervals.view()[4] == 5);
    }

    SECTION("parallelScan_withReturn" + execName)
    {
        NeoFOAM::Field<NeoFOAM::localIdx> intervals(exec, {1, 2, 3, 4, 5});
        NeoFOAM::Field<NeoFOAM::localIdx> segments(exec, intervals.size() + 1, 0);
        auto segView = segments.view();
        const auto intView = intervals.view();
        NeoFOAM::localIdx finalValue = 0;

        NeoFOAM::parallelScan(
            exec,
            {1, segView.size()},
            KOKKOS_LAMBDA(const std::size_t i, NeoFOAM::localIdx& update, const bool final) {
                update += intView[i - 1];
                if (final)
                {
                    segView[i] = update;
                }
            },
            finalValue
        );

        REQUIRE(finalValue == 15);

        auto hostSegments = segments.copyToHost();
        REQUIRE(hostSegments.view()[0] == 0);
        REQUIRE(hostSegments.view()[1] == 1);
        REQUIRE(hostSegments.view()[2] == 3);
        REQUIRE(hostSegments.view()[3] == 6);
        REQUIRE(hostSegments.view()[4] == 10);
        REQUIRE(hostSegments.view()[5] == 15);

        auto hostIntervals = intervals.copyToHost();
        REQUIRE(hostIntervals.view()[0] == 1);
        REQUIRE(hostIntervals.view()[1] == 2);
        REQUIRE(hostIntervals.view()[2] == 3);
        REQUIRE(hostIntervals.view()[3] == 4);
        REQUIRE(hostIntervals.view()[4] == 5);
    }
};
