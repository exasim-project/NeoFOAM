// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

TEST_CASE("Field Constructors")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Copy Constructor " + execName)
    {
        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> a(exec, size);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::Field<NeoFOAM::scalar> b(a);

        REQUIRE(b.size() == size);

        auto hostB = b.copyToHost();
        for (auto value : hostB.view())
        {
            REQUIRE(value == 5.0);
        }

        NeoFOAM::Field<NeoFOAM::scalar> initWith5(exec, size, 5.0);
        REQUIRE(initWith5.size() == size);

        auto hostInitWith5 = initWith5.copyToHost();
        for (auto value : hostInitWith5.view())
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

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("Field Operator+= " + execName)
    {
        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> a(exec, size);
        NeoFOAM::Field<NeoFOAM::scalar> b(exec, size);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        a += b;

        auto hostViewA = a.copyToHost();
        for (auto value : hostViewA.view())
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

        auto hostA = a.copyToHost();
        for (auto value : hostA.view())
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
        auto hostC = c.copyToHost();
        for (auto value : hostC.view())
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

        auto hostC = c.copyToHost();
        for (auto value : hostC.view())
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

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

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

    SECTION("view" + execName)
    {
        NeoFOAM::Field<NeoFOAM::label> a(exec, {1, 2, 3});
        auto hostA = a.copyToHost();

        auto view = hostA.view();
        REQUIRE(view[0] == 1);
        REQUIRE(view[1] == 2);
        REQUIRE(view[2] == 3);

        auto subView = hostA.view({1, 3});
        REQUIRE(subView[0] == 2);
        REQUIRE(subView[1] == 3);
    }

    SECTION("viewVector" + execName)
    {
        NeoFOAM::Field<NeoFOAM::Vector> a(exec, {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}});
        auto hostA = a.copyToHost();

        auto view = hostA.view();
        REQUIRE(view[0] == NeoFOAM::Vector(1, 1, 1));
        REQUIRE(view[1] == NeoFOAM::Vector(2, 2, 2));
        REQUIRE(view[2] == NeoFOAM::Vector(3, 3, 3));

        auto subView = hostA.view({1, 3});
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

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("Field_" + execName)
    {
        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> a(exec, size);
        NeoFOAM::fill(a, 5.0);

        REQUIRE(equal(a, 5.0));

        NeoFOAM::Field<NeoFOAM::scalar> b(exec, size + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.view().size() == size + 2);
        REQUIRE(equal(a, b));

        add(a, b);
        REQUIRE(a.view().size() == size + 2);
        REQUIRE(equal(a, 20.0));

        a = a + b;
        REQUIRE(equal(a, 30.0));

        a = a - b;
        REQUIRE(equal(a, 20.0));

        a = a * 0.1;
        REQUIRE(equal(a, 2.0));

        a = a * b;
        REQUIRE(equal(a, 20.0));

        auto sB = b.view();
        a.apply(KOKKOS_LAMBDA(const NeoFOAM::size_t i) { return 2 * sB[i]; });
        REQUIRE(equal(a, 20.0));
    }
}

TEST_CASE("getSpans")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    NeoFOAM::Field<NeoFOAM::scalar> a(exec, 3, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> b(exec, 3, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> c(exec, 3, 3.0);

    auto [hostA, hostB, hostC] = NeoFOAM::copyToHosts(a, b, c);
    auto [viewB, viewC] = NeoFOAM::spans(b, c);

    REQUIRE(hostA.view()[0] == 1.0);
    REQUIRE(hostB.view()[0] == 2.0);
    REQUIRE(hostC.view()[0] == 3.0);

    NeoFOAM::parallelFor(
        a, KOKKOS_LAMBDA(const NeoFOAM::size_t i) { return viewB[i] + viewC[i]; }
    );

    auto hostD = a.copyToHost();

    for (auto value : hostD.view())
    {
        REQUIRE(value == 5.0);
    }
}
