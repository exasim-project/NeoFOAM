// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include <string>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/linearAlgebra/CSRMatrix.hpp"

TEST_CASE("LinearSystem")
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


    SECTION("Const Entry on " + execName) { sparseMatrixConst.entry() }
}
