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
