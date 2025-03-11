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
        auto csrSparseView = sparseMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                checkSparseSpan[0] = csrSparseView.entry(0, 0);
                checkSparseSpan[1] = csrSparseView.entry(1, 1);
                checkSparseSpan[2] = csrSparseView.entry(1, 2);
                checkSparseSpan[3] = csrSparseView.entry(2, 1);
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
        auto csrDenseView = denseMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                checkDenseSpan[0] = csrDenseView.entry(0, 0);
                checkDenseSpan[1] = csrDenseView.entry(0, 1);
                checkDenseSpan[2] = csrDenseView.entry(0, 2);
                checkDenseSpan[3] = csrDenseView.entry(1, 0);
                checkDenseSpan[4] = csrDenseView.entry(1, 1);
                checkDenseSpan[5] = csrDenseView.entry(1, 2);
                checkDenseSpan[6] = csrDenseView.entry(2, 0);
                checkDenseSpan[7] = csrDenseView.entry(2, 1);
                checkDenseSpan[8] = csrDenseView.entry(2, 2);
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
        auto csrSparseView = sparseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                csrSparseView.entry(0, 0) = -1.0;
                csrSparseView.entry(1, 1) = -5.0;
                csrSparseView.entry(1, 2) = -6.0;
                csrSparseView.entry(2, 1) = -8.0;
            }
        );

        auto hostMatrix = sparseMatrix.copyToHost();
        auto checkHost = hostMatrix.values();
        REQUIRE(checkHost[0] == -1.0);
        REQUIRE(checkHost[1] == -5.0);
        REQUIRE(checkHost[2] == -6.0);
        REQUIRE(checkHost[3] == -8.0);

        // Dense
        auto csrDenseView = denseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                csrDenseView.entry(0, 0) = -1.0;
                csrDenseView.entry(0, 1) = -2.0;
                csrDenseView.entry(0, 2) = -3.0;
                csrDenseView.entry(1, 0) = -4.0;
                csrDenseView.entry(1, 1) = -5.0;
                csrDenseView.entry(1, 2) = -6.0;
                csrDenseView.entry(2, 0) = -7.0;
                csrDenseView.entry(2, 1) = -8.0;
                csrDenseView.entry(2, 2) = -9.0;
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
        auto csrSparseView = sparseMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                checkSparseSpan[0] = csrSparseView.entry(0);
                checkSparseSpan[1] = csrSparseView.entry(1);
                checkSparseSpan[2] = csrSparseView.entry(2);
                checkSparseSpan[3] = csrSparseView.entry(3);
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
        auto csrDenseView = denseMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                checkDenseSpan[0] = csrDenseView.entry(0);
                checkDenseSpan[1] = csrDenseView.entry(1);
                checkDenseSpan[2] = csrDenseView.entry(2);
                checkDenseSpan[3] = csrDenseView.entry(3);
                checkDenseSpan[4] = csrDenseView.entry(4);
                checkDenseSpan[5] = csrDenseView.entry(5);
                checkDenseSpan[6] = csrDenseView.entry(6);
                checkDenseSpan[7] = csrDenseView.entry(7);
                checkDenseSpan[8] = csrDenseView.entry(8);
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
        auto csrSparseView = sparseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                csrSparseView.entry(0) = -1.0;
                csrSparseView.entry(1) = -5.0;
                csrSparseView.entry(2) = -6.0;
                csrSparseView.entry(3) = -8.0;
            }
        );


        auto hostMatrix = sparseMatrix.copyToHost();
        auto checkHost = hostMatrix.values();
        REQUIRE(checkHost[0] == -1.0);
        REQUIRE(checkHost[1] == -5.0);
        REQUIRE(checkHost[2] == -6.0);
        REQUIRE(checkHost[3] == -8.0);

        // Dense
        auto csrDenseView = denseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const size_t) {
                csrDenseView.entry(0) = -1.0;
                csrDenseView.entry(1) = -2.0;
                csrDenseView.entry(2) = -3.0;
                csrDenseView.entry(3) = -4.0;
                csrDenseView.entry(4) = -5.0;
                csrDenseView.entry(5) = -6.0;
                csrDenseView.entry(6) = -7.0;
                csrDenseView.entry(7) = -8.0;
                csrDenseView.entry(8) = -9.0;
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

    SECTION("Span " + execName)
    {
        auto hostMatrix = sparseMatrix.copyToHost();
        auto value = hostMatrix.view().value;
        auto column = hostMatrix.view().columnIndex;
        auto row = hostMatrix.view().rowOffset;
        auto hostvaluesSparse = valuesSparse.copyToHost();
        auto hostcolIdxSparse = colIdxSparse.copyToHost();
        auto hostrowPtrsSparse = rowPtrsSparse.copyToHost();

        REQUIRE(hostvaluesSparse.size() == value.size());
        REQUIRE(hostcolIdxSparse.size() == column.size());
        REQUIRE(hostrowPtrsSparse.size() == row.size());

        for (size_t i = 0; i < value.size(); ++i)
        {
            REQUIRE(hostvaluesSparse.span()[i] == value[i]);
            REQUIRE(hostcolIdxSparse.span()[i] == column[i]);
        }
        for (size_t i = 0; i < row.size(); ++i)
        {
            REQUIRE(hostrowPtrsSparse.span()[i] == row[i]);
        }
    }
}
