// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"

#include "NeoFOAM/linearAlgebra/ginkgo.hpp"
#include "NeoFOAM/linearAlgebra/petsc.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

#if NF_WITH_GINKGO

TEST_CASE("MatrixAssembly - Ginkgo")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    gko::matrix_data<double, int> expected {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};

    SECTION("createMatrix_" + execName)
    {
        NeoFOAM::la::ginkgo::MatrixBuilder<double> builder(exec, {3, 3}, 3 * 3);
        auto kokkosAssembly = builder.startAssembly();
        NeoFOAM::parallelFor(
            exec,
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                kokkosAssembly.insert(i * 3, {i, i - 1, i > 0 ? -1.0 : 0});
                kokkosAssembly.insert(i * 3 + 1, {i, i, 2});
                kokkosAssembly.insert(i * 3 + 2, {i, i + 1, (i < 3 - 1) ? -1.0 : 0});
            }
        );
        NeoFOAM::la::ginkgo::Matrix matrix(std::move(builder));

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
        NeoFOAM::la::ginkgo::MatrixBuilder<double> builder(exec, {3, 3}, 3 * 3);
        auto kokkosAssembly = builder.startAssembly();
        NeoFOAM::parallelFor(
            exec,
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                kokkosAssembly.insert(i * 3, {i, i - 1, i > 0 ? -1.0 : 0});
                kokkosAssembly.insert(i * 3 + 1, {i, i, 2});
                kokkosAssembly.insert(i * 3 + 2, {i, i + 1, (i < 3 - 1) ? -1.0 : 0});
            }
        );
        NeoFOAM::la::ginkgo::Matrix matrix(std::move(builder));
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

#endif

#if NF_WITH_PETSC

TEST_CASE("MatrixAssembly - Petsc")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("assemble_" + execName)
    {
        NeoFOAM::la::petsc::MatrixBuilder builder({3, 3}, 3 * 3);
        auto symAssembly = builder.startSymbolicAssembly();

        // this has to be on a Host executor
        NeoFOAM::parallelFor(
            symAssembly.getCompatibleExecutor(exec),
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                symAssembly.insert(i * 3, {i, i - 1});
                symAssembly.insert(i * 3 + 1, {i, i});
                symAssembly.insert(i * 3 + 2, {i, i + 1});
            }
        );
        auto numAssembly = builder.startNumericAssembly(std::move(symAssembly));
        NeoFOAM::parallelFor(
            exec,
            {0, 3},
            KOKKOS_LAMBDA(const int i) {
                numAssembly.insert(i * 3, i > 0 ? -1.0 : 0);
                numAssembly.insert(i * 3 + 1, 2.0);
                numAssembly.insert(i * 3 + 2, (i < 3 - 1) ? -1.0 : 0);
            }
        );
        NeoFOAM::la::petsc::Matrix mat(std::move(builder));

        // There is no real test here, since I have no idea how to
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

#endif
