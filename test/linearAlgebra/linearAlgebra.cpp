// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"
#include "executorGenerator.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

#if NF_WITH_GINKGO

template<typename ExecSpace>
bool isNotKokkosThreads([[maybe_unused]] ExecSpace ex)
{
#ifdef KOKKOS_ENABLE_THREADS
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Threads>)
    {
        return false;
    }
#endif
    return true;
}

using NeoFOAM::Executor;
using NeoFOAM::Dictionary;
using NeoFOAM::scalar;
using NeoFOAM::localIdx;
using NeoFOAM::Field;
using NeoFOAM::la::LinearSystem;
using NeoFOAM::la::CSRMatrix;
using NeoFOAM::la::ginkgo::Solver;

TEST_CASE("MatrixAssembly - Ginkgo")
{
    Executor exec = GENERATE(allAvailableExecutor());

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    gko::matrix_data<double, int> expected {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};

    SECTION("Solve linear system " + execName)
    {

        Field<scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
        Field<localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
        Field<localIdx> rowPtrs(exec, {0, 3, 6, 9});
        CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowPtrs);

        Field<scalar> rhs(exec, 3, 2.0);
        LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);
        Field<scalar> x(exec, {0.0, 0.0, 0.0});

        Dictionary solverDict {
            {{"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 20}, {"relative_residual_norm", 1e-7}}}}}
        };

        // Create solver
        auto solver = Solver<scalar>(exec, solverDict);

        // Solve system
        solver.solve(linearSystem, x);

        auto hostX = x.copyToHost();
        for (size_t i = 0; i < x.size(); ++i)
        {
            CHECK(hostX[i] != 0.0);
        }
    }
}

#endif
