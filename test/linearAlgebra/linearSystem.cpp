// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <string>

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Field;
using NeoN::la::LinearSystem;
using NeoN::la::CSRMatrix;
using NeoN::la::spmv;

TEST_CASE("LinearSystem")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    Field<scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    Field<localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
    Field<localIdx> rowPtrs(exec, {0, 3, 6, 9});
    CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowPtrs);

    SECTION("construct " + execName)
    {

        Field<scalar> rhs(exec, 3, 0.0);
        LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);

        REQUIRE(linearSystem.matrix().values().size() == 9);
        REQUIRE(linearSystem.matrix().colIdxs().size() == 9);
        REQUIRE(linearSystem.matrix().rowPtrs().size() == 4);
        REQUIRE(linearSystem.matrix().nRows() == 3);
        REQUIRE(linearSystem.rhs().size() == 3);
    }

    SECTION("construct zero initialized from sparsity " + execName)
    {
        auto nCells = 10;
        auto nFaces = 9;
        auto nnz = nCells + 2 * nFaces;
        auto mesh = create1DUniformMesh(exec, nCells);

        // TODO improve structure here
        auto sp = NeoN::finiteVolume::cellCentred::SparsityPattern {mesh};
        auto linearSystem = NeoN::la::createEmptyLinearSystem<
            scalar,
            localIdx,
            NeoN::finiteVolume::cellCentred::SparsityPattern>(sp);

        REQUIRE(linearSystem.matrix().values().size() == nnz);
        REQUIRE(linearSystem.matrix().colIdxs().size() == nnz);
        REQUIRE(linearSystem.matrix().rowPtrs().size() == nCells + 1);
        REQUIRE(linearSystem.matrix().nRows() == nCells);
        REQUIRE(linearSystem.rhs().size() == nCells);
    }

    SECTION("view read/write " + execName)
    {
        Field<scalar> rhs(exec, {10.0, 20.0, 30.0});
        LinearSystem<scalar, localIdx> ls(csrMatrix, rhs);

        auto lsView = ls.view();
        auto hostLS = ls.copyToHost();
        auto hostLSView = hostLS.view();

        // some simple sanity checks
        REQUIRE(hostLSView.A.value.size() == 9);
        REQUIRE(hostLSView.A.columnIndex.size() == 9);
        REQUIRE(hostLSView.A.rowOffset.size() == 4);
        REQUIRE(hostLSView.b.size() == 3);

        // check system values
        for (size_t i = 0; i < hostLSView.A.value.size(); ++i)
        {
            REQUIRE(hostLSView.A.value[i] == static_cast<scalar>(i + 1));
            REQUIRE(hostLSView.A.columnIndex[i] == (i % 3));
        }
        for (size_t i = 0; i < hostLSView.A.rowOffset.size(); ++i)
        {
            REQUIRE(hostLSView.A.rowOffset[i] == static_cast<localIdx>(i * 3));
        }
        for (size_t i = 0; i < hostLSView.b.size(); ++i)
        {
            REQUIRE(hostLSView.b[i] == static_cast<scalar>((i + 1) * 10));
        }

        // Modify values.
        parallelFor(
            exec,
            {0, lsView.A.value.size()},
            KOKKOS_LAMBDA(const size_t i) { lsView.A.value[i] = -lsView.A.value[i]; }
        );

        // Modify values.
        parallelFor(
            exec,
            {0, lsView.b.size()},
            KOKKOS_LAMBDA(const size_t i) { lsView.b[i] = -lsView.b[i]; }
        );

        // Check modification.
        hostLS = ls.copyToHost();
        hostLSView = hostLS.view();
        for (size_t i = 0; i < hostLSView.A.value.size(); ++i)
        {
            REQUIRE(hostLSView.A.value[i] == -static_cast<scalar>(i + 1));
        }
        for (size_t i = 0; i < hostLSView.b.size(); ++i)
        {
            REQUIRE(hostLSView.b[i] == -static_cast<scalar>((i + 1) * 10));
        }
    }


    SECTION("SpmV" + execName)
    {
        Field<scalar> rhs(exec, 3, 0.0);
        LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);
        Field<scalar> x(exec, {1.0, 2.0, 3.0});

        Field<scalar> y = spmv(linearSystem, x);
        auto yHost = y.copyToHost();
        auto yHostView = yHost.span();

        REQUIRE(yHostView[0] == 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0);
        REQUIRE(yHostView[1] == 4.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0);
        REQUIRE(yHostView[2] == 7.0 * 1.0 + 8.0 * 2.0 + 9.0 * 3.0);

        // test with non-zero rhs
        Field<scalar> rhs2(exec, {1.0, 2.0, 3.0});
        LinearSystem<scalar, localIdx> linearSystem2(csrMatrix, rhs2);
        y = spmv(linearSystem2, x);
        yHost = y.copyToHost();
        yHostView = yHost.span();

        REQUIRE(yHostView[0] == 14.0 - 1.0);
        REQUIRE(yHostView[1] == 32.0 - 2.0);
        REQUIRE(yHostView[2] == 50.0 - 3.0);
    }
}
