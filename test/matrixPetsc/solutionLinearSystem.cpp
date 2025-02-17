// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/NeoFOAM.hpp"
#include <Kokkos_Core.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat.h>
#include <petscksp.h>


TEST_CASE("solution linear system")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("ksp" + execName)
    {

        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> values(
            exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
        );
        PetscInt colIdx[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        PetscInt rowIdx[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        PetscScalar v[10];
        PetscScalar w[1];
        // NeoFOAM::Field<PetscInt> colIdx(exec, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        // NeoFOAM::Field<PetscInt> rowIdx(exec, {0, 1, 2, 4, 4, 5, 6, 7, 8, 9});

        Vec x, b, u; /* approx solution, RHS, exact solution */
        Mat A;       /* linear system matrix */
        KSP ksp;     /* linear solver context */
        PC pc;       /* preconditioner context */
        PetscErrorCode ierr;

        PetscInitialize(NULL, NULL, NULL, NULL);


        MatCreate(PETSC_COMM_WORLD, &A);
        MatSetSizes(A, size, size, PETSC_DECIDE, PETSC_DECIDE);
        VecCreate(PETSC_COMM_SELF, &b);
        VecSetSizes(b, PETSC_DECIDE, size);

        std::cout << execName << "\n";
        if (execName == "GPUExecutor")
        {
            VecSetType(b, VECKOKKOS);
            MatSetType(A, MATAIJKOKKOS);
        }
        else
        {
            VecSetType(b, VECSEQ);
            MatSetType(A, MATSEQAIJ);
        }
        VecDuplicate(b, &x);
        VecSetPreallocationCOO(b, size, rowIdx);
        VecSetValuesCOO(b, values.data(), ADD_VALUES);

        MatSetPreallocationCOO(A, size, colIdx, rowIdx);
        MatSetValuesCOO(A, values.data(), ADD_VALUES);


        KSPCreate(PETSC_COMM_WORLD, &ksp);
        KSPSetOperators(ksp, A, A);
        // KSPSetTolerances(ksp, 1.e-9, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
        KSPSetFromOptions(ksp);
        // KSPSetUp(ksp);


        std::cout << "before"
                  << "\n";
        KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
        ierr = KSPSolve(ksp, b, x);
        std::cout << "after"
                  << "\n";
        KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
        VecView(x, PETSC_VIEWER_STDOUT_WORLD);

        VecGetValues(x, size, rowIdx, v);


        REQUIRE(v[0] == 1.);
        REQUIRE(v[1] == 1.);
        REQUIRE(v[2] == 1.);
        REQUIRE(v[3] == 1.);
        REQUIRE(v[4] == 1.);
        REQUIRE(v[5] == 1.);
        REQUIRE(v[6] == 1.);
        REQUIRE(v[7] == 1.);
        REQUIRE(v[8] == 1.);
        REQUIRE(v[9] == 1.);
    }
}
