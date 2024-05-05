// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "NeoFOAM/fields/Field.hpp"

TEST_CASE("Field Operations")
{

    SECTION("CPU")
    {
        int size = 10;
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, size);
        auto sA = a.field();
        NeoFOAM::fill(a, 5.0);

        for (int i = 0; i < size; i++)
        {
            REQUIRE(sA[i] == 5.0);
        }
        NeoFOAM::Field<NeoFOAM::scalar> b(cpuExec, size + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.field().size() == size + 2);

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 10.0);
        }

        add(a, b);
        REQUIRE(a.field().size() == size + 2);

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a + b;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 30.0);
        }

        a = a - b;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a * 0.1;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 2.0);
        }

        a = a * b;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        auto sB = b.field();
        a.apply(KOKKOS_LAMBDA(int i) { return 2 * sB[i]; });

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }
    };

    SECTION("OpenMP")
    {
        int size = 10;
        NeoFOAM::OMPExecutor ompExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(ompExec, size);
        auto sA = a.field();
        NeoFOAM::fill(a, 5.0);

        for (int i = 0; i < size; i++)
        {
            REQUIRE(sA[i] == 5.0);
        }
        NeoFOAM::Field<NeoFOAM::scalar> b(ompExec, size + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.field().size() == size + 2);
        ;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 10.0);
        }

        add(a, b);
        REQUIRE(a.field().size() == size + 2);

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a + b;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 30.0);
        }

        a = a - b;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a * 0.1;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 2.0);
        }

        a = a * b;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        auto sB = b.field();
        a.apply(KOKKOS_LAMBDA(int i) { return 2 * sB[i]; });

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }
    };

    SECTION("GPU")
    {
        int size = 10;
        NeoFOAM::GPUExecutor gpuExec {};
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> gpuA(gpuExec, size);
        NeoFOAM::fill(gpuA, 5.0);

        NeoFOAM::Field<NeoFOAM::scalar> cpuA(cpuExec, size);
        NeoFOAM::fill(cpuA, 10.0);
        for (int i = 0; i < size; i++)
        {
            REQUIRE(cpuA.field()[i] == 10.0);
        }
        cpuA = gpuA.copyToHost();

        for (int i = 0; i < size; i++)
        {
            REQUIRE(cpuA.field()[i] == 5.0);
        }

        NeoFOAM::Field<NeoFOAM::scalar> a(gpuExec, size);
        auto sA = a.field();
        NeoFOAM::fill(a, 5.0);

        NeoFOAM::Field<NeoFOAM::scalar> b(gpuExec, size + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.field().size() == size + 2);

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 10.0);
        }

        add(a, b);
        REQUIRE(a.field().size() == size + 2);

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }

        a = a + b;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 30.0);
        }

        a = a - b;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }

        a = a * 0.1;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 2.0);
        }

        a = a * b;

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }

        auto sB = b.field();
        a.apply(KOKKOS_LAMBDA(int i) { return 2 * sB[i]; });

        for (int i = 0; i < size + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }
    };
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
        };
    };
};
