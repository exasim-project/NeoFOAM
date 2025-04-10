// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

#if NF_WITH_GINKGO

using NeoN::Executor;
using NeoN::Dictionary;
using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Field;
using NeoN::la::LinearSystem;
using NeoN::la::CSRMatrix;
using NeoN::la::ginkgo::Solver;

TEST_CASE("MatrixAssembly - Ginkgo")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    gko::matrix_data<double, int> expected {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};

    SECTION("Solve linear system " + execName)
    {

        Field<scalar> values(exec, {1.0, -0.1, -0.1, 1.0, -0.1, -0.1, 1.0});
        Field<localIdx> colIdx(exec, {0, 1, 0, 1, 2, 1, 2});
        Field<localIdx> rowPtrs(exec, {0, 2, 5, 7});
        CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowPtrs);

        Field<scalar> rhs(exec, {1.0, 2.0, 3.0});
        LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);
        Field<scalar> x(exec, {0.0, 0.0, 0.0});

        Dictionary solverDict {
            {{"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
        };

        // Create solver
        auto solver = Solver<scalar>(exec, solverDict);

        // Solve system
        solver.solve(linearSystem, x);

        auto hostX = x.copyToHost();
        auto hostXS = hostX.span();
        REQUIRE((hostXS[0]) == Catch::Approx(1.24489796).margin(1e-8));
        REQUIRE((hostXS[1]) == Catch::Approx(2.44897959).margin(1e-8));
        REQUIRE((hostXS[2]) == Catch::Approx(3.24489796).margin(1e-8));
    }
}

#endif
