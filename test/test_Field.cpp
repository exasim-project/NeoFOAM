// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "NeoFOAM/fields/Field.hpp"
#include "NeoFOAM/fields/FieldOperations.hpp"
#include "NeoFOAM/fields/FieldTypeDefs.hpp"

int main(int argc, char* argv[])
{

    // Initialize Catch2
    Kokkos::initialize(argc, argv);
    Catch::Session session;

    // Specify command line options
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    int result = session.run();

    // Run benchmarks if there are any
    Kokkos::finalize();

    return result;
}

TEST_CASE("Field Operations")
{

    SECTION("CPU")
    {
        int N = 10;
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, N);
        auto s_a = a.field();
        NeoFOAM::fill(a, 5.0);

        for (int i = 0; i < N; i++)
        {
            REQUIRE(s_a[i] == 5.0);
        }
        NeoFOAM::Field<NeoFOAM::scalar> b(cpuExec, N + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.field().size() == N + 2);

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 10.0);
        }

        add(a, b);
        REQUIRE(a.field().size() == N + 2);

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a + b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 30.0);
        }

        a = a - b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a * 0.1;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 2.0);
        }

        a = a * b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        auto s_b = b.field();
        a.apply(KOKKOS_LAMBDA(int i) { return 2 * s_b[i]; });

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }
    };

    SECTION("OpenMP")
    {
        int N = 10;
        NeoFOAM::OMPExecutor OMPExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(OMPExec, N);
        auto s_a = a.field();
        NeoFOAM::fill(a, 5.0);

        for (int i = 0; i < N; i++)
        {
            REQUIRE(s_a[i] == 5.0);
        }
        NeoFOAM::Field<NeoFOAM::scalar> b(OMPExec, N + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.field().size() == N + 2);
        ;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 10.0);
        }

        add(a, b);
        REQUIRE(a.field().size() == N + 2);

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a + b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 30.0);
        }

        a = a - b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a * 0.1;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 2.0);
        }

        a = a * b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        auto s_b = b.field();
        a.apply(KOKKOS_LAMBDA(int i) { return 2 * s_b[i]; });

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }
    };

    SECTION("GPU")
    {
        int N = 10;
        NeoFOAM::GPUExecutor gpuExec {};
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> GPUa(gpuExec, N);
        NeoFOAM::fill(GPUa, 5.0);

        NeoFOAM::Field<NeoFOAM::scalar> CPUa(cpuExec, N);
        NeoFOAM::fill(CPUa, 10.0);
        for (int i = 0; i < N; i++)
        {
            REQUIRE(CPUa.field()[i] == 10.0);
        }
        CPUa = GPUa.copyToHost();

        for (int i = 0; i < N; i++)
        {
            REQUIRE(CPUa.field()[i] == 5.0);
        }

        NeoFOAM::Field<NeoFOAM::scalar> a(gpuExec, N);
        auto s_a = a.field();
        NeoFOAM::fill(a, 5.0);

        NeoFOAM::Field<NeoFOAM::scalar> b(gpuExec, N + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.field().size() == N + 2);

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 10.0);
        }

        add(a, b);
        REQUIRE(a.field().size() == N + 2);

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }

        a = a + b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 30.0);
        }

        a = a - b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }

        a = a * 0.1;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 2.0);
        }

        a = a * b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }

        auto s_b = b.field();
        a.apply(KOKKOS_LAMBDA(int i) { return 2 * s_b[i]; });

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }
    };
}
