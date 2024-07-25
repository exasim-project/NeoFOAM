// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/linearAlgebra/linearAlgebra.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

TEST_CASE("MatrixAssembly")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    gko::matrix_data<double, int> expected {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};


    SECTION("assemble_" + execName)
    {
        auto kokkosAssembly = NeoFOAM::GinkgoMatrixAssembly<double, int>(exec, 3, 3, 3 * 3);

        NeoFOAM::parallelFor(
            exec,
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                if (i > 0)
                {
                    kokkosAssembly.insert(i * 3, {i, i - 1, -1.0});
                }
                kokkosAssembly.insert(i * 3 + 1, {i, i, 2});
                if (i < 3 - 1)
                {
                    kokkosAssembly.insert(i * 3 + 2, {i, i + 1, -1.0});
                }
            }
        );
        kokkosAssembly.finalize();
        auto& gkoAssembly = kokkosAssembly.getUnderlyingData();

        auto hostGkoAssembly = gkoAssembly.copy_to_host();
        REQUIRE(hostGkoAssembly.nonzeros.size() == expected.nonzeros.size());
        for (size_t i = 0; i < expected.nonzeros.size(); ++i)
        {
            CHECK(hostGkoAssembly.nonzeros[i] == expected.nonzeros[i]);
        }
    }

    SECTION("createMatrix_" + execName)
    {
        auto kokkosAssembly = NeoFOAM::GinkgoMatrixAssembly<double, int>(exec, 3, 3, 3 * 3);
        NeoFOAM::parallelFor(
            exec,
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                if (i > 0)
                {
                    kokkosAssembly.insert(i * 3, {i, i - 1, -1.0});
                }
                kokkosAssembly.insert(i * 3 + 1, {i, i, 2});
                if (i < 3 - 1)
                {
                    kokkosAssembly.insert(i * 3 + 2, {i, i + 1, -1.0});
                }
            }
        );

        NeoFOAM::Matrix<double> matrix(std::move(kokkosAssembly));
        gko::matrix_data<double, int> hostGkoAssembly;
        matrix.getUnderlyingData()->write(hostGkoAssembly);
        REQUIRE(hostGkoAssembly.nonzeros.size() == expected.nonzeros.size());
        for (size_t i = 0; i < expected.nonzeros.size(); ++i)
        {
            CHECK(hostGkoAssembly.nonzeros[i] == expected.nonzeros[i]);
        }
    }

    SECTION("applyMatrix_" + execName)
    {
        auto kokkosAssembly = NeoFOAM::GinkgoMatrixAssembly<double, int>(exec, 3, 3, 3 * 3);
        NeoFOAM::parallelFor(
            exec,
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                if (i > 0)
                {
                    kokkosAssembly.insert(i * 3, {i, i - 1, -1.0});
                }
                kokkosAssembly.insert(i * 3 + 1, {i, i, 2});
                if (i < 3 - 1)
                {
                    kokkosAssembly.insert(i * 3 + 2, {i, i + 1, -1.0});
                }
            }
        );
        NeoFOAM::Matrix<double> matrix(std::move(kokkosAssembly));
        NeoFOAM::Field<double> in(exec, {1, 1, 1});
        NeoFOAM::Field<double> out(exec, {0, 0, 0});
        NeoFOAM::Field<double> expected(NeoFOAM::SerialExecutor {}, {1, 0, 1});

        matrix.apply(in, out);

        auto hostOut = out.copyToHost();
        REQUIRE(hostOut.size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
        {
            CHECK(hostOut[i] == expected[i]);
        }
    }
}
