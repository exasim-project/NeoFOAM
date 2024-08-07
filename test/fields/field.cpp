// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/fields/fieldTypeDefs.hpp"
#include "NeoFOAM/fields/operations/operationsMacros.hpp"
#include "NeoFOAM/fields/operations/comparison.hpp"

TEST_CASE("Field Constructors")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("Copy Constructor " + execName)
    {
        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> a(exec, size);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::Field<NeoFOAM::scalar> b(a);

        REQUIRE(b.size() == size);

        auto hostSpanB = b.copyToHost().span();
        for (auto value : hostSpanB)
        {
            REQUIRE(value == 5.0);
        }
    }

    SECTION("Initialiser List Constructor " + execName)
    {
        NeoFOAM::Field<NeoFOAM::label> a(exec, {1, 2, 3});
        auto hostA = a.copyToHost();
        REQUIRE(hostA.data()[0] == 1);
        REQUIRE(hostA.data()[1] == 2);
        REQUIRE(hostA.data()[2] == 3);
    }

    SECTION("Cross Exec Constructor " + execName)
    {

        NeoFOAM::Field<NeoFOAM::label> a(exec, {1, 2, 3});
        NeoFOAM::Field<NeoFOAM::label> b(a);

        auto hostB = b.copyToHost();

        REQUIRE(hostB.data()[0] == 1);
        REQUIRE(hostB.data()[1] == 2);
        REQUIRE(hostB.data()[2] == 3);
    }
}

TEST_CASE("Field Operator Overloads")
{

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("Field Operator+= " + execName)
    {
        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> a(exec, size);
        NeoFOAM::Field<NeoFOAM::scalar> b(exec, size);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        a += b;

        auto hostSpanA = a.copyToHost().span();
        for (auto value : hostSpanA)
        {
            REQUIRE(value == 15.0);
        }
    }

    SECTION("Field Operator-= " + execName)
    {
        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> a(exec, size);
        NeoFOAM::Field<NeoFOAM::scalar> b(exec, size);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        a -= b;

        auto hostSpanA = a.copyToHost().span();
        for (auto value : hostSpanA)
        {
            REQUIRE(value == -5.0);
        }
    }

    SECTION("Field Operator+ " + execName)
    {
        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> a(exec, size);
        NeoFOAM::Field<NeoFOAM::scalar> b(exec, size);
        NeoFOAM::Field<NeoFOAM::scalar> c(exec, size);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        c = a + b;
        auto hostSpanC = c.copyToHost().span();
        for (auto value : hostSpanC)
        {
            REQUIRE(value == 15.0);
        }
    }

    SECTION("Field Operator-" + execName)
    {
        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> a(exec, size);
        NeoFOAM::Field<NeoFOAM::scalar> b(exec, size);
        NeoFOAM::Field<NeoFOAM::scalar> c(exec, size);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        c = a - b;

        auto hostSpanC = c.copyToHost().span();
        for (auto value : hostSpanC)
        {
            REQUIRE(value == -5.0);
        }
    }
}

TEST_CASE("Field Container Operations")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("empty, size, range" + execName)
    {
        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> a(exec, 0);
        NeoFOAM::Field<NeoFOAM::scalar> b(exec, size);
        REQUIRE(a.empty() == true);
        REQUIRE(a.size() == 0);
        REQUIRE(a.range().first == 0);
        REQUIRE(a.range().second == 0);
        REQUIRE(b.empty() == false);
        REQUIRE(b.size() == size);
        REQUIRE(b.range().first == 0);
        REQUIRE(b.range().second == size);
    };

    SECTION("span" + execName)
    {
        NeoFOAM::Field<NeoFOAM::label> a(exec, {1, 2, 3});
        auto hostA = a.copyToHost();

        auto view = hostA.span();
        REQUIRE(view[0] == 1);
        REQUIRE(view[1] == 2);
        REQUIRE(view[2] == 3);

        auto subView = hostA.span({1, 2});
        REQUIRE(subView[0] == 2);
        REQUIRE(subView[1] == 3);
    }

    SECTION("spanVector" + execName)
    {
        NeoFOAM::Field<NeoFOAM::Vector> a(exec, {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}});
        auto hostA = a.copyToHost();

        auto view = hostA.span();
        REQUIRE(view[0] == NeoFOAM::Vector(1, 1, 1));
        REQUIRE(view[1] == NeoFOAM::Vector(2, 2, 2));
        REQUIRE(view[2] == NeoFOAM::Vector(3, 3, 3));

        auto subView = hostA.span({1, 2});
        REQUIRE(subView[0] == NeoFOAM::Vector(2, 2, 2));
        REQUIRE(subView[1] == NeoFOAM::Vector(3, 3, 3));
    }

    SECTION("copyToHost " + execName)
    {

        NeoFOAM::Field<NeoFOAM::label> a(exec, {1, 2, 3});

        auto hostA = a.copyToHost();
        auto hostB = a.copyToHost();

        REQUIRE(&(hostA.data()[0]) != &(hostB.data()[0]));
        REQUIRE(hostA.data()[1] == hostB.data()[1]);
    }
}

TEST_CASE("Field Operations")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("Field_" + execName)
    {
        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> a(exec, size);
        auto sA = a.span();
        NeoFOAM::fill(a, 5.0);

        REQUIRE(equal(a, 5.0));

        NeoFOAM::Field<NeoFOAM::scalar> b(exec, size + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.span().size() == size + 2);
        REQUIRE(equal(a, b));

        add(a, b);
        REQUIRE(a.span().size() == size + 2);
        REQUIRE(equal(a, 20.0));

        a = a + b;
        REQUIRE(equal(a, 30.0));

        a = a - b;
        REQUIRE(equal(a, 20.0));

        a = a * 0.1;
        REQUIRE(equal(a, 2.0));

        a = a * b;
        REQUIRE(equal(a, 20.0));

        auto sB = b.span();
        a.apply(KOKKOS_LAMBDA(const NeoFOAM::size_t i) { return 2 * sB[i]; });
        REQUIRE(equal(a, 20.0));
    }
}
