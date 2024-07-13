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

TEST_CASE("Field Copy Constructor")
{
    SECTION("CPU")
    {
        int N = 10;
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, N);
        NeoFOAM::fill(a, 5.0);

        NeoFOAM::Field<NeoFOAM::scalar> b(a);

        REQUIRE(b.size() == N);

        for (int i = 0; i < N; i++)
        {
            REQUIRE(b.span()[i] == 5.0);
        }
    };

    SECTION("OpenMP")
    {
        int N = 10;
        NeoFOAM::OMPExecutor OMPExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(OMPExec, N);
        NeoFOAM::fill(a, 5.0);

        NeoFOAM::Field<NeoFOAM::scalar> b(a);

        REQUIRE(b.size() == N);

        for (int i = 0; i < N; i++)
        {
            REQUIRE(b.span()[i] == 5.0);
        }
    };

    SECTION("GPU")
    {
        int N = 10;
        NeoFOAM::GPUExecutor gpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(gpuExec, N);
        NeoFOAM::fill(a, 5.0);

        NeoFOAM::Field<NeoFOAM::scalar> b(a);

        REQUIRE(b.size() == N);

       auto hostB = b.copyToHost();
        for (auto value: hostB.span())
        {
            REQUIRE( value == 5.0);
        }
    };
}

//--------------------------------------------------------------------------------
// Assignment Operators
//--------------------------------------------------------------------------------

TEST_CASE("Field Operator+=")
{
    SECTION("CPU")
    {
        int N = 10;
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(cpuExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        a += b;

        for (int i = 0; i < N; i++)
        {
            REQUIRE(a.span()[i] == 15.0);
        }
    };

    SECTION("OpenMP")
    {
        int N = 10;
        NeoFOAM::OMPExecutor OMPExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(OMPExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(OMPExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        a += b;

        for (int i = 0; i < N; i++)
        {
            REQUIRE(a.span()[i] == 15.0);
        }
    };

    SECTION("GPU")
    {
        int N = 10;
        NeoFOAM::GPUExecutor gpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(gpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(gpuExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        a += b;

        for (int i = 0; i < N; i++)
        {
            REQUIRE(a.copyToHost().span()[i] == 15.0);
        }
    };
}

TEST_CASE("Field Operator-=")
{
    SECTION("CPU")
    {
        int N = 10;
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(cpuExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        a -= b;

        for (int i = 0; i < N; i++)
        {
            REQUIRE(a.span()[i] == -5.0);
        }
    };

    SECTION("OpenMP")
    {
        int N = 10;
        NeoFOAM::OMPExecutor OMPExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(OMPExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(OMPExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        a -= b;

        for (int i = 0; i < N; i++)
        {
            REQUIRE(a.span()[i] == -5.0);
        }
    };

    SECTION("GPU")
    {
        int N = 10;
        NeoFOAM::GPUExecutor gpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(gpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(gpuExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        a -= b;

        for (int i = 0; i < N; i++)
        {
            REQUIRE(a.copyToHost().span()[i] == -5.0);
        }
    };
}

TEST_CASE("Field Operator+")
{
    SECTION("CPU")
    {
        int N = 10;
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(cpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> c(cpuExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        c = a + b;
        for (int i = 0; i < N; i++)
        {
            REQUIRE(c.span()[i] == 15.0);
        }
    };

    SECTION("OpenMP")
    {
        int N = 10;
        NeoFOAM::OMPExecutor OMPExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(OMPExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(OMPExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> c(OMPExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        c = a + b;

        for (int i = 0; i < N; i++)
        {
            REQUIRE(c.span()[i] == 15.0);
        }
    };

    SECTION("GPU")
    {
        int N = 10;
        NeoFOAM::GPUExecutor gpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(gpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(gpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> c(gpuExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        c = a + b;

        for (int i = 0; i < N; i++)
        {
            REQUIRE(c.span()[i] == 15.0);
        }
    };
}

TEST_CASE("Field Operator-")
{
    SECTION("CPU")
    {
        int N = 10;
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(cpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> c(cpuExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        c = a - b;

        for (int i = 0; i < N; i++)
        {
            REQUIRE(c.span()[i] == -5.0);
        }
    };

    SECTION("OpenMP")
    {
        int N = 10;
        NeoFOAM::OMPExecutor OMPExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(OMPExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(OMPExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> c(OMPExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        c = a - b;

        for (int i = 0; i < N; i++)
        {
            REQUIRE(c.span()[i] == -5.0);
        }
    };

    SECTION("GPU")
    {
        int N = 10;
        NeoFOAM::GPUExecutor gpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(gpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> b(gpuExec, N);
        NeoFOAM::Field<NeoFOAM::scalar> c(gpuExec, N);
        NeoFOAM::fill(a, 5.0);
        NeoFOAM::fill(b, 10.0);

        c = a - b;

        for (int i = 0; i < N; i++)
        {
            REQUIRE(c.span()[i] == -5.0);
        }
    };
}

TEST_CASE("Field empty, size, range")
{
    SECTION("CPU")
    {
        int N = 10;
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, 0);
        NeoFOAM::Field<NeoFOAM::scalar> b(cpuExec, N);
        REQUIRE(a.empty() == true);
        REQUIRE(a.size() == 0);
        REQUIRE(a.range().first == 0);
        REQUIRE(a.range().second == N);
        REQUIRE(b.empty() == false);
        REQUIRE(b.size() == N);
        REQUIRE(b.range().first == 0);
        REQUIRE(b.range().second == N);
    };

    SECTION("OpenMP")
    {
        int N = 10;
        NeoFOAM::OMPExecutor OMPExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(OMPExec, 0);
        NeoFOAM::Field<NeoFOAM::scalar> b(OMPExec, N);
        REQUIRE(a.empty() == true);
        REQUIRE(a.size() == 0);
        REQUIRE(a.range().first == 0);
        REQUIRE(a.range().second == N);
        REQUIRE(b.empty() == false);
        REQUIRE(b.size() == N);
        REQUIRE(b.range().first == 0);
        REQUIRE(b.range().second == N);
    };

    SECTION("GPU")
    {
        int N = 10;
        NeoFOAM::GPUExecutor gpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(gpuExec, 0);
        NeoFOAM::Field<NeoFOAM::scalar> b(gpuExec, N);
        REQUIRE(a.empty() == true);
        REQUIRE(a.size() == 0);
        REQUIRE(a.range().first == 0);
        REQUIRE(a.range().second == N);
        REQUIRE(b.empty() == false);
        REQUIRE(b.size() == N);
        REQUIRE(b.range().first == 0);
        REQUIRE(b.range().second == N);
    };
}

TEST_CASE("Field Operations")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::OMPExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("Can initialize Field from initializer list on " + execName)
    {

        NeoFOAM::Field<NeoFOAM::label> a(exec, {1, 2, 3});

        auto hostA = a.copyToHost();

        REQUIRE(hostA.data()[0] == 1);
        REQUIRE(hostA.data()[1] == 2);
        REQUIRE(hostA.data()[2] == 3);
    }

    SECTION("Can initialize Field from a Field on " + execName)
    {

        NeoFOAM::Field<NeoFOAM::label> a(exec, {1, 2, 3});
        NeoFOAM::Field<NeoFOAM::label> b(a);

        auto hostB = b.copyToHost();

        REQUIRE(hostB.data()[0] == 1);
        REQUIRE(hostB.data()[1] == 2);
        REQUIRE(hostB.data()[2] == 3);
    }

    SECTION("Can create a subview " + execName)
    {

        NeoFOAM::Field<NeoFOAM::label> a(exec, {1, 2, 3});

        auto hostA = a.copyToHost();
        auto subView = hostA.span({1, 2});

        REQUIRE(subView[0] == 2);
        REQUIRE(subView[1] == 3);
    }

    SECTION("Copy to host creates a copy from " + execName)
    {

        NeoFOAM::Field<NeoFOAM::label> a(exec, {1, 2, 3});

        auto hostA = a.copyToHost();
        auto hostB = a.copyToHost();

        hostA.data()[0] = 0;

        REQUIRE(hostA.data()[0] != hostB.data()[0]);
        REQUIRE(hostA.data()[1] == hostB.data()[1]);
    }

    SECTION("Can set via a subview " + execName)
    {

        NeoFOAM::Field<NeoFOAM::label> a(exec, {1, 2, 3});

        auto hostA = a.copyToHost();
        auto subView = hostA.span({1, 2});
        subView[0] = 5;

        REQUIRE(subView[0] == 5);
        REQUIRE(subView[1] == 3);
    }

    SECTION("Field_" + execName)
    {
        int size = 10;
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

        auto s_b = b.span();
        a.apply(KOKKOS_LAMBDA(int i) { return 2 * s_b[i]; });
        REQUIRE(equal(a, 20.0));
    }
}

TEST_CASE("Primitives")
{
    SECTION("Vector")
    {
        SECTION("CPU")
        {
            NeoFOAM::Vector a(1.0, 2.0, 3.0);
            REQUIRE(a(0) == 1.0);
            REQUIRE(a(1) == 2.0);
            REQUIRE(a(2) == 3.0);

            NeoFOAM::Vector b(1.0, 2.0, 3.0);
            REQUIRE(a == b);

            NeoFOAM::Vector c(2.0, 4.0, 6.0);

            REQUIRE(a + b == c);

            REQUIRE((a - b) == NeoFOAM::Vector(0.0, 0.0, 0.0));

            a += b;
            REQUIRE(a == c);

            a -= b;
            REQUIRE(a == b);
            a *= 2;
            REQUIRE(a == c);
            a = b;

            REQUIRE(a == b);

            NeoFOAM::Vector d(4.0, 8.0, 12.0);
            REQUIRE((a + a + a + a) == d);
            REQUIRE((4 * a) == d);
            REQUIRE((a * 4) == d);
            REQUIRE((a + 3 * a) == d);
            REQUIRE((a + 2 * a + a) == d);
        }
    }
}
