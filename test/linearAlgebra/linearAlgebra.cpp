// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"

#define KOKKOS_ENABLE_SERIAL

#include "NeoFOAM/linearAlgebra/ginkgo.hpp"
#include "NeoFOAM/linearAlgebra/CSRMatrix.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

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
        NeoFOAM::la::LinearSystem<NeoFOAM::scalar, int> linearSystem(csrMatrix, rhs);
        NeoFOAM::Field<NeoFOAM::scalar> x(exec, {0.0, 0.0, 0.0});

        size_t nrows = linearSystem.rhs().size();
        auto gkoExec = NeoFOAM::la::ginkgo::getGkoExecutor(exec);
        auto gkoRhs = gko::matrix::Dense<NeoFOAM::scalar>::create(
            gkoExec,
            gko::dim<2> {nrows, 1},
            gko::array<NeoFOAM::scalar>::view(gkoExec, nrows, linearSystem.rhs().data()),
            1
        );
        auto gkoX = gko::matrix::Dense<NeoFOAM::scalar>::create(
            gkoExec,
            gko::dim<2> {nrows, 1},
            gko::array<NeoFOAM::scalar>::view(gkoExec, nrows, x.data()),
            1
        );

        auto valuesView = gko::array<NeoFOAM::scalar>::view(gkoExec, values.size(), values.data());
        auto colIdxView = gko::array<int>::view(gkoExec, colIdx.size(), colIdx.data());
        auto rowPtrView = gko::array<int>::view(gkoExec, rowPtrs.size(), rowPtrs.data());

        auto gkoA = gko::share(gko::matrix::Csr<NeoFOAM::scalar, int>::create(
            gkoExec, gko::dim<2> {nrows, nrows}, valuesView, colIdxView, rowPtrView
        ));

        // Create solver factory
        auto solver_gen =
            gko::solver::Cg<NeoFOAM::scalar>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(20u),
                    gko::stop::ResidualNorm<NeoFOAM::scalar>::build().with_reduction_factor(1e-07)
                )
                // Add preconditioner, these 2 lines are the only
                // difference from the simple solver example
                // .with_preconditioner(bj::build().with_max_block_size(8u))
                .on(gkoExec);
        // Create solver
        auto solver = solver_gen->generate(gkoA);

        // // Solve system
        solver->apply(gkoRhs, gkoX);

        auto hostX = x.copyToHost();
        for (size_t i = 0; i < nrows; ++i)
        {
            CHECK(hostX[i] != 0.0);
        }
    }
}

#endif
