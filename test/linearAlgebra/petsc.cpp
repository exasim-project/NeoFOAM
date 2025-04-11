// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"
#include <catch2/catch_approx.hpp>


#define KOKKOS_ENABLE_SERIAL

#include "NeoFOAM/NeoFOAM.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

#if NF_WITH_PETSC

using NeoFOAM::Executor;
using NeoFOAM::Dictionary;
using NeoFOAM::scalar;
using NeoFOAM::localIdx;
using NeoFOAM::Field;
using NeoFOAM::la::LinearSystem;
using NeoFOAM::la::CSRMatrix;
using NeoFOAM::la::Solver;

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

        Field<NeoFOAM::scalar> values(exec, {10.0, 4.0, 7.0, 2.0, 10.0, 8.0, 3.0, 6.0, 10.0});
        // TODO work on support for unsingned types
        Field<localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
        Field<localIdx> rowPtrs(exec, {0, 3, 6, 9});
        CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowPtrs);

        Field<NeoFOAM::scalar> rhs(exec, {1.0, 2.0, 3.0});
        LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);
        Field<NeoFOAM::scalar> x(exec, {0.0, 0.0, 0.0});

        NeoFOAM::Dictionary solverDict {{
            {"solver", std::string {"Petsc"}},
        }};

        // Create solver
        auto solver = NeoFOAM::la::Solver(exec, solverDict);

        // Solve system
        solver.solve(linearSystem, x);

        auto hostX = x.copyToHost();
        auto hostXS = hostX.span();
        REQUIRE((hostXS[0]) == Catch::Approx(3. / 205.).margin(1e-8));
        REQUIRE((hostXS[1]) == Catch::Approx(8. / 205.).margin(1e-8));
        REQUIRE((hostXS[2]) == Catch::Approx(53. / 205.).margin(1e-8));


        SECTION("Solve linear system second time " + execName)
        {
            // NeoFOAM::Database db;
            NeoFOAM::Field<NeoFOAM::scalar> values(
                exec, {10.0, 2.0, 3.0, 5.0, 20.0, 2.0, 4.0, 4.0, 30.0}
            );

            Field<localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
            Field<localIdx> rowPtrs(exec, {0, 3, 6, 9});
            CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowPtrs);

            Field<NeoFOAM::scalar> rhs(exec, {1.0, 2.0, 3.0});
            LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);
            Field<NeoFOAM::scalar> x(exec, {0.0, 0.0, 0.0});

            NeoFOAM::Dictionary solverDict {{
                {"solver", std::string {"Petsc"}},
            }};

            // Create solver
            auto solver = NeoFOAM::la::Solver(exec, solverDict);

            // Solve system
            solver.solve(linearSystem, x);

            auto hostX = x.copyToHost();
            auto hostXS = hostX.span();
            REQUIRE((hostXS[0]) == Catch::Approx(8. / 341.).margin(1e-8));
            REQUIRE((hostXS[1]) == Catch::Approx(27. / 341.).margin(1e-8));
            REQUIRE((hostXS[2]) == Catch::Approx(63. / 682.).margin(1e-8));
        }
    }
}

#endif
