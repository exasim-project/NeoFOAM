// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("Field Constructors")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Copy Constructor " + execName)
    {
        NeoN::size_t size = 10;
        NeoN::Field<NeoN::scalar> a(exec, size);
        NeoN::fill(a, 5.0);
        NeoN::Field<NeoN::scalar> b(a);

        REQUIRE(b.size() == size);

        auto hostB = b.copyToHost();
        for (auto value : hostB.span())
        {
            REQUIRE(value == 5.0);
        }

        NeoN::Field<NeoN::scalar> initWith5(exec, size, 5.0);
        REQUIRE(initWith5.size() == size);

        auto hostInitWith5 = initWith5.copyToHost();
        for (auto value : hostInitWith5.span())
        {
            REQUIRE(value == 5.0);
        }
    }

    SECTION("Initialiser List Constructor " + execName)
    {
        NeoN::Field<NeoN::label> a(exec, {1, 2, 3});
        auto hostA = a.copyToHost();
        REQUIRE(hostA.data()[0] == 1);
        REQUIRE(hostA.data()[1] == 2);
        REQUIRE(hostA.data()[2] == 3);
    }

    SECTION("Cross Exec Constructor " + execName)
    {

        NeoN::Field<NeoN::label> a(exec, {1, 2, 3});
        NeoN::Field<NeoN::label> b(a);

        auto hostB = b.copyToHost();

        REQUIRE(hostB.data()[0] == 1);
        REQUIRE(hostB.data()[1] == 2);
        REQUIRE(hostB.data()[2] == 3);
    }
}


TEST_CASE("Field Operator Overloads")
{

    NeoN::Executor exec = GENERATE(
        NeoN::Executor(NeoN::SerialExecutor {}),
        NeoN::Executor(NeoN::CPUExecutor {}),
        NeoN::Executor(NeoN::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("Field Operator+= " + execName)
    {
        NeoN::size_t size = 10;
        NeoN::Field<NeoN::scalar> a(exec, size);
        NeoN::Field<NeoN::scalar> b(exec, size);
        NeoN::fill(a, 5.0);
        NeoN::fill(b, 10.0);

        a += b;

        auto hostSpanA = a.copyToHost();
        for (auto value : hostSpanA.span())
        {
            REQUIRE(value == 15.0);
        }
    }

    SECTION("Field Operator-= " + execName)
    {
        NeoN::size_t size = 10;
        NeoN::Field<NeoN::scalar> a(exec, size);
        NeoN::Field<NeoN::scalar> b(exec, size);
        NeoN::fill(a, 5.0);
        NeoN::fill(b, 10.0);

        a -= b;

        auto hostA = a.copyToHost();
        for (auto value : hostA.span())
        {
            REQUIRE(value == -5.0);
        }
    }

    SECTION("Field Operator+ " + execName)
    {
        NeoN::size_t size = 10;
        NeoN::Field<NeoN::scalar> a(exec, size);
        NeoN::Field<NeoN::scalar> b(exec, size);
        NeoN::Field<NeoN::scalar> c(exec, size);
        NeoN::fill(a, 5.0);
        NeoN::fill(b, 10.0);

        c = a + b;
        auto hostC = c.copyToHost();
        for (auto value : hostC.span())
        {
            REQUIRE(value == 15.0);
        }
    }

    SECTION("Field Operator-" + execName)
    {
        NeoN::size_t size = 10;
        NeoN::Field<NeoN::scalar> a(exec, size);
        NeoN::Field<NeoN::scalar> b(exec, size);
        NeoN::Field<NeoN::scalar> c(exec, size);
        NeoN::fill(a, 5.0);
        NeoN::fill(b, 10.0);

        c = a - b;

        auto hostC = c.copyToHost();
        for (auto value : hostC.span())
        {
            REQUIRE(value == -5.0);
        }
    }
}

TEST_CASE("Field Container Operations")
{
    NeoN::Executor exec = GENERATE(
        NeoN::Executor(NeoN::SerialExecutor {}),
        NeoN::Executor(NeoN::CPUExecutor {}),
        NeoN::Executor(NeoN::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("empty, size, range" + execName)
    {
        NeoN::size_t size = 10;
        NeoN::Field<NeoN::scalar> a(exec, 0);
        NeoN::Field<NeoN::scalar> b(exec, size);
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
        NeoN::Field<NeoN::label> a(exec, {1, 2, 3});
        auto hostA = a.copyToHost();

        auto view = hostA.span();
        REQUIRE(view[0] == 1);
        REQUIRE(view[1] == 2);
        REQUIRE(view[2] == 3);

        auto subView = hostA.span({1, 3});
        REQUIRE(subView[0] == 2);
        REQUIRE(subView[1] == 3);
    }

    SECTION("spanVector" + execName)
    {
        NeoN::Field<NeoN::Vector> a(exec, {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}});
        auto hostA = a.copyToHost();

        auto view = hostA.span();
        REQUIRE(view[0] == NeoN::Vector(1, 1, 1));
        REQUIRE(view[1] == NeoN::Vector(2, 2, 2));
        REQUIRE(view[2] == NeoN::Vector(3, 3, 3));

        auto subView = hostA.span({1, 3});
        REQUIRE(subView[0] == NeoN::Vector(2, 2, 2));
        REQUIRE(subView[1] == NeoN::Vector(3, 3, 3));
    }

    SECTION("copyToHost " + execName)
    {

        NeoN::Field<NeoN::label> a(exec, {1, 2, 3});

        auto hostA = a.copyToHost();
        auto hostB = a.copyToHost();

        REQUIRE(&(hostA.data()[0]) != &(hostB.data()[0]));
        REQUIRE(hostA.data()[1] == hostB.data()[1]);
    }
}

TEST_CASE("Field Operations")
{
    NeoN::Executor exec = GENERATE(
        NeoN::Executor(NeoN::SerialExecutor {}),
        NeoN::Executor(NeoN::CPUExecutor {}),
        NeoN::Executor(NeoN::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("Field_" + execName)
    {
        NeoN::size_t size = 10;
        NeoN::Field<NeoN::scalar> a(exec, size);
        NeoN::fill(a, 5.0);

        REQUIRE(equal(a, 5.0));

        NeoN::Field<NeoN::scalar> b(exec, size + 2);
        NeoN::fill(b, 10.0);

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
        a.apply(KOKKOS_LAMBDA(const NeoN::size_t i) { return 2 * sB[i]; });
        REQUIRE(equal(a, 20.0));
    }
}

TEST_CASE("getSpans")
{
    NeoN::Executor exec = GENERATE(
        NeoN::Executor(NeoN::SerialExecutor {}),
        NeoN::Executor(NeoN::CPUExecutor {}),
        NeoN::Executor(NeoN::GPUExecutor {})
    );

    NeoN::Field<NeoN::scalar> a(exec, 3, 1.0);
    NeoN::Field<NeoN::scalar> b(exec, 3, 2.0);
    NeoN::Field<NeoN::scalar> c(exec, 3, 3.0);

    auto [hostA, hostB, hostC] = NeoN::copyToHosts(a, b, c);
    auto [spanB, spanC] = NeoN::spans(b, c);

    REQUIRE(hostA.span()[0] == 1.0);
    REQUIRE(hostB.span()[0] == 2.0);
    REQUIRE(hostC.span()[0] == 3.0);

    NeoN::parallelFor(
        a, KOKKOS_LAMBDA(const NeoN::size_t i) { return spanB[i] + spanC[i]; }
    );

    auto hostD = a.copyToHost();

    for (auto value : hostD.span())
    {
        REQUIRE(value == 5.0);
    }
}
