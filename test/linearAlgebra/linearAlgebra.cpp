// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"

#define KOKKOS_ENABLE_SERIAL

#include "NeoFOAM/NeoFOAM.hpp"

#if NF_WITH_GINKGO

template<typename ExecSpace>
bool isNotKokkosThreads(ExecSpace ex)
{
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Threads>)
    {
        return false;
    }
    return true;
}

TEST_CASE("MatrixAssembly - Ginkgo")
{


    // NOTE: Ginkgo doesn't support Kokkos::Threads, the only option is to use omp threads
    // thus we need to filter out all executors which underlying executor is Kokkos::Threads
    // TODO: This seems to be a very convoluted approach, hopefully there is a better approach
    NeoFOAM::Executor exec = GENERATE(filter(
        [](auto exec)
        { return std::visit([](auto e) { return isNotKokkosThreads(e.underlyingExec()); }, exec); },
        values(
            {NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
             NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
             NeoFOAM::Executor(NeoFOAM::GPUExecutor {})}
        )
    ));


    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    gko::matrix_data<double, int> expected {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};

    SECTION("Solve linear system " + execName)
    {

        NeoFOAM::Field<NeoFOAM::scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
        // TODO work on support for unsingned types
        NeoFOAM::Field<int> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
        NeoFOAM::Field<int> rowPtrs(exec, {0, 3, 6, 9});
        NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, int> csrMatrix(values, colIdx, rowPtrs);

        NeoFOAM::Field<NeoFOAM::scalar> rhs(exec, 3, 2.0);
        NeoFOAM::la::LinearSystem<NeoFOAM::scalar, int> linearSystem(csrMatrix, rhs, "custom");
        NeoFOAM::Field<NeoFOAM::scalar> x(exec, {0.0, 0.0, 0.0});

        NeoFOAM::Dictionary solverDict;
        solverDict.insert("maxIters", 100);
        solverDict.insert("relTol", float(1e-7));

        // Create solver
        auto solver = NeoFOAM::la::ginkgo::CG<NeoFOAM::scalar>(exec, solverDict);

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
