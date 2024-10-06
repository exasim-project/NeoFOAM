// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"


template<typename SpanA, typename SpanB, typename T>
struct kernelLambda_1
{
    SpanA spanA;
    SpanB spanB;
    T val;

    KOKKOS_FUNCTION void operator()(const size_t i) const { spanA[i] = spanB[i] + val; }
};

template<typename Span, typename T>
struct kernelLambda_2
{
    Span span;
    T val;

    KOKKOS_FUNCTION T operator()(const size_t i) const { return span[i] + val; }
};


template<typename Span>
struct kernelLambda_3
{
    Span span;

    KOKKOS_FUNCTION void operator()(const size_t i, double& lsum) const { lsum += span[i]; }
};

TEST_CASE("parallelFor")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("parallelFor_" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto spanA = fieldA.span();
        auto spanB = fieldB.span();
        NeoFOAM::fill(fieldB, 1.0);

        NeoFOAM::scalar val = 2.0;
        kernelLambda_1<decltype(spanA), decltype(spanB), decltype(val)> kernel {spanA, spanB, val};
        NeoFOAM::parallelFor(exec, {0, 5}, kernel);
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

        NeoFOAM::Vector val = NeoFOAM::Vector(2.0, 2.0, 2.0);
        kernelLambda_1<decltype(spanA), decltype(spanB), decltype(val)> kernel {spanA, spanB, val};
        NeoFOAM::parallelFor(exec, {0, 5}, kernel);
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

        NeoFOAM::scalar val = 2.0;
        kernelLambda_2<decltype(spanB), decltype(val)> kernel {spanB, val};
        NeoFOAM::parallelFor(fieldA, kernel);
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
    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("parallelReduce_" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto spanB = fieldB.span();
        NeoFOAM::fill(fieldB, 1.0);
        NeoFOAM::scalar sum = 0.0;

        kernelLambda_3<decltype(spanB)> kernel {spanB};
        NeoFOAM::parallelReduce(exec, {0, 5}, kernel, sum);

        REQUIRE(sum == 5.0);
    }

    SECTION("parallelReduce_Field_" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 5);
        NeoFOAM::fill(fieldA, 0.0);
        NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 5);
        auto spanB = fieldB.span();
        NeoFOAM::fill(fieldB, 1.0);
        NeoFOAM::scalar sum = 0.0;

        kernelLambda_3<decltype(spanB)> kernel {spanB};
        NeoFOAM::parallelReduce(fieldA, kernel, sum);

        REQUIRE(sum == 5.0);
    }
};
