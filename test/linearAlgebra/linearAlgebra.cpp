// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"

#define KOKKOS_ENABLE_SERIAL

#include "NeoFOAM/linearAlgebra/ginkgo.hpp"
#include "NeoFOAM/linearAlgebra/petsc.hpp"
#include "NeoFOAM/linearAlgebra/CSRMatrix.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

#if NF_WITH_GINKGO

TEST_CASE("MatrixAssembly - Ginkgo")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}) //,
                                                      // NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
                                                      // NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

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

    SECTION("Solve linear system " + execName)
    {

        NeoFOAM::Field<NeoFOAM::scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
        // TODO work on support for unsingned types
        NeoFOAM::Field<int> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
        NeoFOAM::Field<int> rowPtrs(exec, {0, 3, 6, 9});
        NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, int> csrMatrix(values, colIdx, rowPtrs);

        NeoFOAM::Field<NeoFOAM::scalar> rhs(exec, 3, 2.0);
        NeoFOAM::la::LinearSystem<NeoFOAM::scalar, int> linearSystem(csrMatrix, rhs);
        NeoFOAM::Field<NeoFOAM::scalar> x(exec, {0.0, 0.0, 0.0});

        size_t nrows = linearSystem.rhs().size();
        auto gkoExec = NeoFOAM::la::ginkgo::getGkoExecutor(exec);
        auto gkoRhs = gko::matrix::Dense<NeoFOAM::scalar>::create(
            gkoExec,
            gko::dim<2> {nrows, 1},
            gko::array<NeoFOAM::scalar>::view(gkoExec, nrows, linearSystem.rhs().data()),
            1
        );
        auto gkoX = gko::matrix::Dense<NeoFOAM::scalar>::create(
            gkoExec,
            gko::dim<2> {nrows, 1},
            gko::array<NeoFOAM::scalar>::view(gkoExec, nrows, x.data()),
            1
        );

        auto valuesView = gko::array<NeoFOAM::scalar>::view(gkoExec, values.size(), values.data());
        auto colIdxView = gko::array<int>::view(gkoExec, colIdx.size(), colIdx.data());
        auto rowPtrView = gko::array<int>::view(gkoExec, rowPtrs.size(), rowPtrs.data());

        auto gkoA = gko::share(gko::matrix::Csr<NeoFOAM::scalar, int>::create(
            gkoExec, gko::dim<2> {nrows, nrows}, valuesView, colIdxView, rowPtrView
        ));

        // Create solver factory
        auto solver_gen =
            gko::solver::Cg<NeoFOAM::scalar>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(20u),
                    gko::stop::ResidualNorm<NeoFOAM::scalar>::build().with_reduction_factor(1e-07)
                )
                // Add preconditioner, these 2 lines are the only
                // difference from the simple solver example
                // .with_preconditioner(bj::build().with_max_block_size(8u))
                .on(gkoExec);
        // Create solver
        auto solver = solver_gen->generate(gkoA);

        // // Solve system
        solver->apply(gkoRhs, gkoX);

        NeoFOAM::Field<NeoFOAM::scalar> outX(exec, gkoX->get_const_values(), nrows);
        auto hostX = outX.copyToHost();
        for (size_t i = 0; i < nrows; ++i)
        {
            CHECK(hostX[i] != 0.0);
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
