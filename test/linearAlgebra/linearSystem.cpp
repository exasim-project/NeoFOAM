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
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}) //,
                                                      // NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
                                                      // NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
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
        REQUIRE(linearSystem.matrix().nNonZeros() == 9);
        REQUIRE(linearSystem.matrix().rowPtrs().size() == 4);
        REQUIRE(linearSystem.matrix().nRows() == 3);

        REQUIRE(linearSystem.rhs().size() == 3);
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
