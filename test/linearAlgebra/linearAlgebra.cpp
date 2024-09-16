// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"

#include "NeoFOAM/linearAlgebra/ginkgo.hpp"
#include "NeoFOAM/linearAlgebra/petsc.hpp"
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
        NeoFOAM::la::ginkgo::Matrix<double> matrix(exec, {3, 3}, 3 * 3);
        auto kokkosAssembly = matrix.startAssembly();

        NeoFOAM::parallelFor(
            exec,
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                kokkosAssembly.insert(i * 3, {i, i - 1, i > 0 ? -1.0 : 0});
                kokkosAssembly.insert(i * 3 + 1, {i, i, 2});
                kokkosAssembly.insert(i * 3 + 2, {i, i + 1, (i < 3 - 1) ? -1.0 : 0});
            }
        );
        matrix.finishAssembly(std::move(kokkosAssembly));

        gko::matrix_data<double, int> hostGkoAssembly;
        matrix.getUnderlyingData()->write(hostGkoAssembly);
        REQUIRE(hostGkoAssembly.nonzeros.size() == expected.nonzeros.size());
        for (size_t i = 0; i < expected.nonzeros.size(); ++i)
        {
            CHECK(hostGkoAssembly.nonzeros[i] == expected.nonzeros[i]);
        }
    }

    SECTION("createMatrix_" + execName)
    {
        NeoFOAM::la::ginkgo::Matrix<double> matrix(exec, {3, 3}, 3 * 3);
        auto kokkosAssembly = matrix.startAssembly();
        NeoFOAM::parallelFor(
            exec,
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                kokkosAssembly.insert(i * 3, {i, i - 1, i > 0 ? -1.0 : 0});
                kokkosAssembly.insert(i * 3 + 1, {i, i, 2});
                kokkosAssembly.insert(i * 3 + 2, {i, i + 1, (i < 3 - 1) ? -1.0 : 0});
            }
        );

        matrix.finishAssembly(std::move(kokkosAssembly));
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
        NeoFOAM::la::ginkgo::Matrix<double> matrix(exec, {3, 3}, 3 * 3);
        auto kokkosAssembly = matrix.startAssembly();
        NeoFOAM::parallelFor(
            exec,
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                kokkosAssembly.insert(i * 3, {i, i - 1, i > 0 ? -1.0 : 0});
                kokkosAssembly.insert(i * 3 + 1, {i, i, 2});
                kokkosAssembly.insert(i * 3 + 2, {i, i + 1, (i < 3 - 1) ? -1.0 : 0});
            }
        );
        matrix.finishAssembly(std::move(kokkosAssembly));
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

TEST_CASE("Petsc")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("assemble_" + execName)
    {
        NeoFOAM::la::petsc::Matrix mat({3, 3}, 3 * 3);
        auto symAssembly = mat.startSymbolicAssembly();

        // this has to be on a Host executor
        NeoFOAM::parallelFor(
            symAssembly.getCompatibleExecutor(exec),
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                if (i > 0)
                {
                    symAssembly.insert(i * 3, {i, i - 1});
                }
                symAssembly.insert(i * 3 + 1, {i, i});
                if (i < 3 - 1)
                {
                    symAssembly.insert(i * 3 + 2, {i, i + 1});
                }
            }
        );
        auto numAssembly = mat.startNumericAssembly(std::move(symAssembly));
        NeoFOAM::parallelFor(
            exec,
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                if (i > 0)
                {
                    numAssembly.insert(i * 3, -1.0);
                }
                numAssembly.insert(i * 3 + 1, 2.0);
                if (i < 3 - 1)
                {
                    numAssembly.insert(i * 3 + 2, -1.0);
                }
            }
        );
        mat.finishNumericAssembly(std::move(numAssembly));

        MatView(mat.getUnderlying(), PETSC_VIEWER_STDOUT_WORLD);
    }

    SECTION("apply_" + execName)
    {
        NeoFOAM::la::petsc::Matrix mat(
            {3, 3},
            {NeoFOAM::CPUExecutor {}, {0, 0, 1, 1, 1, 2, 2}},
            {NeoFOAM::CPUExecutor {}, {0, 1, 0, 1, 2, 1, 2}},
            {exec, {2, -1, -1, 2, -1, -1, 2}}
        );

        NeoFOAM::Field<double> in(exec, {1, 1, 1});
        NeoFOAM::Field<double> out(exec, {0, 0, 0});
        NeoFOAM::Field<double> expected(NeoFOAM::SerialExecutor {}, {1, 0, 1});

        mat.apply(in, out);

        auto hostOut = out.copyToHost();
        REQUIRE(hostOut.size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
        {
            CHECK(hostOut[i] == expected[i]);
        }
    }
}
