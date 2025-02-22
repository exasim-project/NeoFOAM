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

    // empty matrix
    NeoFOAM::Field<NeoFOAM::scalar> valuesEmpty(exec, {});
    NeoFOAM::Field<NeoFOAM::localIdx> colIdxEmpty(exec, {});
    NeoFOAM::Field<NeoFOAM::localIdx> rowPtrsEmpty(exec, {});
    NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> emptyMatrix(
        valuesEmpty, colIdxEmpty, rowPtrsEmpty
    );
    NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> emptyMatrixConst(
        valuesEmpty, colIdxEmpty, rowPtrsEmpty
    );

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
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t i) {
                checkSparseSpan[0] = sparseMatrixConst.entry(0, 0);
                checkSparseSpan[1] = sparseMatrixConst.entry(1, 1);
                checkSparseSpan[2] = sparseMatrixConst.entry(1, 2);
                checkSparseSpan[3] = sparseMatrixConst.entry(2, 1);
            }
        );

        auto checkHost = checkSparse.copyToHost();
        REQUIRE(checkHost.span()[0] == 1.0);
        REQUIRE(checkHost.span()[1] == 5.0);
        REQUIRE(checkHost.span()[2] == 6.0);
        REQUIRE(checkHost.span()[3] == 8.0);

        // dense
        NeoFOAM::Field<NeoFOAM::scalar> checkDense(exec, 9);
        auto checkDenseSpan = checkDense.span();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t i) {
                checkDenseSpan[0] = denseMatrixConst.entry(0, 0);
                checkDenseSpan[1] = denseMatrixConst.entry(0, 1);
                checkDenseSpan[2] = denseMatrixConst.entry(0, 2);
                checkDenseSpan[3] = denseMatrixConst.entry(1, 0);
                checkDenseSpan[4] = denseMatrixConst.entry(1, 1);
                checkDenseSpan[5] = denseMatrixConst.entry(1, 2);
                checkDenseSpan[6] = denseMatrixConst.entry(2, 0);
                checkDenseSpan[7] = denseMatrixConst.entry(2, 1);
                checkDenseSpan[8] = denseMatrixConst.entry(2, 2);
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
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t i) {
                sparseMatrix.entry(0, 0) = -1.0;
                sparseMatrix.entry(1, 1) = -5.0;
            }
        );

        // auto checkHost = checkSparse.copyToHost();
        // REQUIRE(checkHost.span()[0] == 1.0);
        // REQUIRE(checkHost.span()[1] == 5.0);
        // REQUIRE(checkHost.span()[2] == 6.0);
        // REQUIRE(checkHost.span()[3] == 8.0);
    }
}
