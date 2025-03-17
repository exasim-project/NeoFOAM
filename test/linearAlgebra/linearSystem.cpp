// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <string>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/linearAlgebra/linearSystem.hpp"

TEST_CASE("LinearSystem")
{

    // FIXME: fix this new generate
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("construct " + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
        NeoFOAM::Field<NeoFOAM::localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
        NeoFOAM::Field<NeoFOAM::localIdx> rowPtrs(exec, {0, 3, 6, 9});
        NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> csrMatrix(
            values, colIdx, rowPtrs
        );

        NeoFOAM::Field<NeoFOAM::scalar> rhs(exec, 3, 0.0);
        NeoFOAM::la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> linearSystem(
            csrMatrix, rhs, "custom"
        );

        REQUIRE(linearSystem.matrix().values().size() == 9);
        REQUIRE(linearSystem.matrix().colIdxs().size() == 9);
        REQUIRE(linearSystem.matrix().rowPtrs().size() == 4);
        REQUIRE(linearSystem.matrix().nRows() == 3);
        REQUIRE(linearSystem.rhs().size() == 3);
    }


    SECTION("view read/write " + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
        NeoFOAM::Field<NeoFOAM::localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
        NeoFOAM::Field<NeoFOAM::localIdx> rowPtrs(exec, {0, 3, 6, 9});
        NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> csrMatrix(
            values, colIdx, rowPtrs
        );
        NeoFOAM::Field<NeoFOAM::scalar> rhs(exec, {10.0, 20.0, 30.0});
        NeoFOAM::la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> ls(csrMatrix, rhs, "custom");

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
            REQUIRE(hostLSView.A.value[i] == static_cast<NeoFOAM::scalar>(i + 1));
            REQUIRE(hostLSView.A.columnIndex[i] == (i % 3));
        }
        for (size_t i = 0; i < hostLSView.A.rowOffset.size(); ++i)
        {
            REQUIRE(hostLSView.A.rowOffset[i] == static_cast<NeoFOAM::localIdx>(i * 3));
        }
        for (size_t i = 0; i < hostLSView.b.size(); ++i)
        {
            REQUIRE(hostLSView.b[i] == static_cast<NeoFOAM::scalar>((i + 1) * 10));
        }

        // Modify values.
        parallelFor(
            exec,
            {0, lsView.A.value.size()},
            KOKKOS_LAMBDA(const size_t i) {
                lsView.A.value[i] = -lsView.A.value[i];
                lsView.b[i] = -lsView.b[i];
            }
        );

        // Check modification.
        hostLS = ls.copyToHost();
        hostLSView = hostLS.view();
        for (size_t i = 0; i < hostLSView.A.value.size(); ++i)
        {
            REQUIRE(hostLSView.A.value[i] == -static_cast<NeoFOAM::scalar>(i + 1));
        }
        for (size_t i = 0; i < hostLSView.b.size(); ++i)
        {
            REQUIRE(hostLSView.b[i] == -static_cast<NeoFOAM::scalar>((i + 1) * 10));
        }
    }


    SECTION("SpmV" + execName)
    {
        NeoFOAM::Field<NeoFOAM::scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
        NeoFOAM::Field<NeoFOAM::localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
        NeoFOAM::Field<NeoFOAM::localIdx> rowPtrs(exec, {0, 3, 6, 9});
        NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> csrMatrix(
            values, colIdx, rowPtrs
        );

        NeoFOAM::Field<NeoFOAM::scalar> rhs(exec, 3, 0.0);
        NeoFOAM::la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> linearSystem(
            csrMatrix, rhs, "testing"
        );
        NeoFOAM::Field<NeoFOAM::scalar> x(exec, {1.0, 2.0, 3.0});

        NeoFOAM::Field<NeoFOAM::scalar> y = NeoFOAM::la::SpMV(linearSystem, x);
        auto yHost = y.copyToHost();

        REQUIRE(yHost[0] == 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0);
        REQUIRE(yHost[1] == 4.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0);
        REQUIRE(yHost[2] == 7.0 * 1.0 + 8.0 * 2.0 + 9.0 * 3.0);


        // test with non-zero rhs
        NeoFOAM::Field<NeoFOAM::scalar> rhs2(exec, {1.0, 2.0, 3.0});
        NeoFOAM::la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> linearSystem2(
            csrMatrix, rhs2, "testing"
        );
        y = NeoFOAM::la::SpMV(linearSystem2, x);
        yHost = y.copyToHost();

        REQUIRE(yHost[0] == 14.0 - 1.0);
        REQUIRE(yHost[1] == 32.0 - 2.0);
        REQUIRE(yHost[2] == 50.0 - 3.0);
    }
}
