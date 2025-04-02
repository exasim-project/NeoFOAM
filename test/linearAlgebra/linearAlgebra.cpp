// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"
#include <catch2/catch_approx.hpp>


#define KOKKOS_ENABLE_SERIAL

#include "NeoFOAM/NeoFOAM.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

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

#if NF_WITH_PETSC

TEST_CASE("MatrixAssembly - Petsc")
{


    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);


    SECTION("Solve linear system " + execName)
    {

        NeoFOAM::Database db;

        std::cout << execName << "\n";
        NeoFOAM::Field<NeoFOAM::scalar> values(
            exec, {10.0, 4.0, 7.0, 2.0, 10.0, 8.0, 3.0, 6.0, 10.0}
        );
        // TODO work on support for unsingned types
        NeoFOAM::Field<int> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
        NeoFOAM::Field<int> rowPtrs(exec, {0, 3, 6, 9});
        NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, int> csrMatrix(values, colIdx, rowPtrs);

        NeoFOAM::Field<NeoFOAM::scalar> rhs(exec, {1.0, 2.0, 3.0});
        NeoFOAM::la::LinearSystem<NeoFOAM::scalar, int> linearSystem(csrMatrix, rhs, "custom");
        NeoFOAM::Field<NeoFOAM::scalar> x(exec, {0.0, 0.0, 0.0});

        NeoFOAM::Dictionary solverDict;

        // Create solver
        auto solver = NeoFOAM::la::petscSolver::petscSolver<NeoFOAM::scalar>(exec, solverDict, db);

        // Solve system
        solver.solve(linearSystem, x);

        auto hostX = x.copyToHost();

        REQUIRE(hostX[0] == Catch::Approx(3. / 205.).margin(1e-6));
        REQUIRE(hostX[1] == Catch::Approx(8. / 205.).margin(1e-6));
        REQUIRE(hostX[2] == Catch::Approx(53. / 205.).margin(1e-6));
    }

    SECTION("Linear solver context " + execName)
    {


        Mat Amat_;
        Vec sol_, rhs_;

        std::cout << execName << "\n";
        NeoFOAM::Field<NeoFOAM::scalar> values(
            exec, {10.0, 4.0, 7.0, 2.0, 10.0, 8.0, 3.0, 6.0, 10.0}
        );
        // TODO work on support for unsingned types
        NeoFOAM::Field<int> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
        NeoFOAM::Field<int> rowPtrs(exec, {0, 3, 6, 9});
        NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, int> csrMatrix(values, colIdx, rowPtrs);

        NeoFOAM::Field<NeoFOAM::scalar> rhs(exec, {1.0, 2.0, 3.0});
        NeoFOAM::la::LinearSystem<NeoFOAM::scalar, int> linearSystem(csrMatrix, rhs, "custom");
        NeoFOAM::Field<NeoFOAM::scalar> x(exec, {0.0, 0.0, 0.0});

        NeoFOAM::la::petscSolverContext::petscSolverContext<NeoFOAM::scalar> petsctx(exec);
        NeoFOAM::Database db;
        // NeoFOAM::Document doc({ {"petsctx", petsctx } });

        // NeoFOAM::CollectionMixin<NeoFOAM::Document>  petscSolverCollection(db,
        // "testpetscContext");

        // db.insert("testpetscContext", petscSolverCollection);

        // auto& testpetscContext =
        // db.at<NeoFOAM::CollectionMixin<NeoFOAM::Document>>("testpetscContext");


        std::size_t nrows = linearSystem.rhs().size();
        petsctx.initialize(linearSystem);

        Amat_ = petsctx.AMat();
        rhs_ = petsctx.rhs();
        sol_ = petsctx.sol();

        VecSetValuesCOO(rhs_, linearSystem.rhs().data(), ADD_VALUES);
        MatSetValuesCOO(Amat_, linearSystem.matrix().values().data(), ADD_VALUES);
    }
}

#endif
