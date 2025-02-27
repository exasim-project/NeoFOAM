// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <Kokkos_Core.hpp>

#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/linearAlgebra/CSRMatrix.hpp"

TEST_CASE("CSRMatrix")
{

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    // sparse matrix
    NeoFOAM::Field<NeoFOAM::scalar> valuesSparse(exec, {1.0, 5.0, 6.0, 8.0});
    NeoFOAM::Field<NeoFOAM::localIdx> colIdxSparse(exec, {0, 1, 2, 1});
    NeoFOAM::Field<NeoFOAM::localIdx> rowPtrsSparse(exec, {0, 1, 3, 4});
    NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> sparseMatrix(
        valuesSparse, colIdxSparse, rowPtrsSparse
    );
    const NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> sparseMatrixConst(
        valuesSparse, colIdxSparse, rowPtrsSparse
    );

    // dense matrix
    NeoFOAM::Field<NeoFOAM::scalar> valuesDense(
        exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
    );
    NeoFOAM::Field<NeoFOAM::localIdx> colIdxDense(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
    NeoFOAM::Field<NeoFOAM::localIdx> rowPtrsDense(exec, {0, 3, 6, 9});
    NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> denseMatrix(
        valuesDense, colIdxDense, rowPtrsDense
    );
    const NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> denseMatrixConst(
        valuesDense, colIdxDense, rowPtrsDense
    );


    SECTION("Read entry on " + execName)
    {
        // Sparse
        NeoFOAM::Field<NeoFOAM::scalar> checkSparse(exec, 4);
        auto checkSparseSpan = checkSparse.span();
        auto csrSparseSpan = sparseMatrixConst.span();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                checkSparseSpan[0] = csrSparseSpan.entry(0, 0);
                checkSparseSpan[1] = csrSparseSpan.entry(1, 1);
                checkSparseSpan[2] = csrSparseSpan.entry(1, 2);
                checkSparseSpan[3] = csrSparseSpan.entry(2, 1);
            }
        );

        auto checkHost = checkSparse.copyToHost();
        REQUIRE(checkHost.span()[0] == 1.0);
        REQUIRE(checkHost.span()[1] == 5.0);
        REQUIRE(checkHost.span()[2] == 6.0);
        REQUIRE(checkHost.span()[3] == 8.0);

        // Dense
        NeoFOAM::Field<NeoFOAM::scalar> checkDense(exec, 9);
        auto checkDenseSpan = checkDense.span();
        auto csrDenseSpan = denseMatrixConst.span();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                checkDenseSpan[0] = csrDenseSpan.entry(0, 0);
                checkDenseSpan[1] = csrDenseSpan.entry(0, 1);
                checkDenseSpan[2] = csrDenseSpan.entry(0, 2);
                checkDenseSpan[3] = csrDenseSpan.entry(1, 0);
                checkDenseSpan[4] = csrDenseSpan.entry(1, 1);
                checkDenseSpan[5] = csrDenseSpan.entry(1, 2);
                checkDenseSpan[6] = csrDenseSpan.entry(2, 0);
                checkDenseSpan[7] = csrDenseSpan.entry(2, 1);
                checkDenseSpan[8] = csrDenseSpan.entry(2, 2);
            }
        );
        checkHost = checkDense.copyToHost();
        REQUIRE(checkHost.span()[0] == 1.0);
        REQUIRE(checkHost.span()[1] == 2.0);
        REQUIRE(checkHost.span()[2] == 3.0);
        REQUIRE(checkHost.span()[3] == 4.0);
        REQUIRE(checkHost.span()[4] == 5.0);
        REQUIRE(checkHost.span()[5] == 6.0);
        REQUIRE(checkHost.span()[6] == 7.0);
        REQUIRE(checkHost.span()[7] == 8.0);
        REQUIRE(checkHost.span()[8] == 9.0);
    }

    SECTION("Update existing entry on " + execName)
    {
        // Sparse
        auto csrSparseSpan = sparseMatrix.span();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                csrSparseSpan.entry(0, 0) = -1.0;
                csrSparseSpan.entry(1, 1) = -5.0;
                csrSparseSpan.entry(1, 2) = -6.0;
                csrSparseSpan.entry(2, 1) = -8.0;
            }
        );

        auto hostMatrix = sparseMatrix.copyToHost();
        auto checkHost = hostMatrix.values();
        REQUIRE(checkHost[0] == -1.0);
        REQUIRE(checkHost[1] == -5.0);
        REQUIRE(checkHost[2] == -6.0);
        REQUIRE(checkHost[3] == -8.0);

        // Dense
        auto csrDenseSpan = denseMatrix.span();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                csrDenseSpan.entry(0, 0) = -1.0;
                csrDenseSpan.entry(0, 1) = -2.0;
                csrDenseSpan.entry(0, 2) = -3.0;
                csrDenseSpan.entry(1, 0) = -4.0;
                csrDenseSpan.entry(1, 1) = -5.0;
                csrDenseSpan.entry(1, 2) = -6.0;
                csrDenseSpan.entry(2, 0) = -7.0;
                csrDenseSpan.entry(2, 1) = -8.0;
                csrDenseSpan.entry(2, 2) = -9.0;
            }
        );

        hostMatrix = denseMatrix.copyToHost();
        checkHost = hostMatrix.values();
        REQUIRE(checkHost[0] == -1.0);
        REQUIRE(checkHost[1] == -2.0);
        REQUIRE(checkHost[2] == -3.0);
        REQUIRE(checkHost[3] == -4.0);
        REQUIRE(checkHost[4] == -5.0);
        REQUIRE(checkHost[5] == -6.0);
        REQUIRE(checkHost[6] == -7.0);
        REQUIRE(checkHost[7] == -8.0);
        REQUIRE(checkHost[8] == -9.0);
    }

    SECTION("Read directValue on " + execName)
    {
        // Sparse
        NeoFOAM::Field<NeoFOAM::scalar> checkSparse(exec, 4);
        auto checkSparseSpan = checkSparse.span();
        auto csrSparseSpan = sparseMatrixConst.span();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                checkSparseSpan[0] = csrSparseSpan.directValue(0);
                checkSparseSpan[1] = csrSparseSpan.directValue(1);
                checkSparseSpan[2] = csrSparseSpan.directValue(2);
                checkSparseSpan[3] = csrSparseSpan.directValue(3);
            }
        );
        auto checkHost = checkSparse.copyToHost();
        REQUIRE(checkHost.span()[0] == 1.0);
        REQUIRE(checkHost.span()[1] == 5.0);
        REQUIRE(checkHost.span()[2] == 6.0);
        REQUIRE(checkHost.span()[3] == 8.0);

        // Dense
        NeoFOAM::Field<NeoFOAM::scalar> checkDense(exec, 9);
        auto checkDenseSpan = checkDense.span();
        auto csrDenseSpan = denseMatrixConst.span();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                checkDenseSpan[0] = csrDenseSpan.directValue(0);
                checkDenseSpan[1] = csrDenseSpan.directValue(1);
                checkDenseSpan[2] = csrDenseSpan.directValue(2);
                checkDenseSpan[3] = csrDenseSpan.directValue(3);
                checkDenseSpan[4] = csrDenseSpan.directValue(4);
                checkDenseSpan[5] = csrDenseSpan.directValue(5);
                checkDenseSpan[6] = csrDenseSpan.directValue(6);
                checkDenseSpan[7] = csrDenseSpan.directValue(7);
                checkDenseSpan[8] = csrDenseSpan.directValue(8);
            }
        );
        checkHost = checkDense.copyToHost();
        REQUIRE(checkHost.span()[0] == 1.0);
        REQUIRE(checkHost.span()[1] == 2.0);
        REQUIRE(checkHost.span()[2] == 3.0);
        REQUIRE(checkHost.span()[3] == 4.0);
        REQUIRE(checkHost.span()[4] == 5.0);
        REQUIRE(checkHost.span()[5] == 6.0);
        REQUIRE(checkHost.span()[6] == 7.0);
        REQUIRE(checkHost.span()[7] == 8.0);
        REQUIRE(checkHost.span()[8] == 9.0);
    }

    SECTION("Update existing directValue on " + execName)
    {
        // Sparse
        auto csrSparseSpan = sparseMatrix.span();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                csrSparseSpan.directValue(0) = -1.0;
                csrSparseSpan.directValue(1) = -5.0;
                csrSparseSpan.directValue(2) = -6.0;
                csrSparseSpan.directValue(3) = -8.0;
            }
        );


        auto hostMatrix = sparseMatrix.copyToHost();
        auto checkHost = hostMatrix.values();
        REQUIRE(checkHost[0] == -1.0);
        REQUIRE(checkHost[1] == -5.0);
        REQUIRE(checkHost[2] == -6.0);
        REQUIRE(checkHost[3] == -8.0);

        // Dense
        auto csrDenseSpan = denseMatrix.span();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                csrDenseSpan.directValue(0) = -1.0;
                csrDenseSpan.directValue(1) = -2.0;
                csrDenseSpan.directValue(2) = -3.0;
                csrDenseSpan.directValue(3) = -4.0;
                csrDenseSpan.directValue(4) = -5.0;
                csrDenseSpan.directValue(5) = -6.0;
                csrDenseSpan.directValue(6) = -7.0;
                csrDenseSpan.directValue(7) = -8.0;
                csrDenseSpan.directValue(8) = -9.0;
            }
        );

        hostMatrix = denseMatrix.copyToHost();
        checkHost = hostMatrix.values();
        REQUIRE(checkHost[0] == -1.0);
        REQUIRE(checkHost[1] == -2.0);
        REQUIRE(checkHost[2] == -3.0);
        REQUIRE(checkHost[3] == -4.0);
        REQUIRE(checkHost[4] == -5.0);
        REQUIRE(checkHost[5] == -6.0);
        REQUIRE(checkHost[6] == -7.0);
        REQUIRE(checkHost[7] == -8.0);
        REQUIRE(checkHost[8] == -9.0);
    }
}
