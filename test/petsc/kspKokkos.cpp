// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
// base on https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/ksp/tutorials/ex1.c?ref_type=heads

static char help[] = "Solves a tridiagonal linear system with KSP.\n\n";

#define PetscTestCheck(expr)                                                                       \
    PetscCheck(                                                                                    \
        expr, PETSC_COMM_SELF, PETSC_ERR_LIB, "Assertion: `%s' failed.", PetscStringize(expr)      \
    )


/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h - vectors
     petscmat.h    - matrices              petscpc.h  - preconditioners
     petscis.h     - index sets
     petscviewer.h - viewers

  Note:  The corresponding parallel example is ex23.c
*/

#include <Kokkos_Core.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat.h>
#include <petscksp.h>

using DefaultMemorySpace = Kokkos::Cuda::memory_space;

PETSC_EXTERN PetscErrorCode FillMatrixKokkosCOO(PetscInt, Mat);
PETSC_EXTERN PetscErrorCode FillVectorKokkosCOO(PetscInt, Vec);

PetscErrorCode FillMatrixKokkosCOO(PetscInt n, Mat A)
{
    // using exec = Kokkos::DefaultHostExecutionSpace;

    Kokkos::View<PetscScalar*, DefaultMemorySpace> v("v", n);

    PetscFunctionBeginUser;
    // Simulation of GPU based finite assembly process with COO
    Kokkos::parallel_for(
        "AssembleElementMatrices", n, KOKKOS_LAMBDA(PetscInt i) { v[i] = float(i) + 1.0; }
    );

    MatSetValuesCOO(A, v.data(), INSERT_VALUES);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FillVectorKokkosCOO(PetscInt n, Vec b)
{
    // using exec = Kokkos::DefaultHostExecutionSpace;

    Kokkos::View<PetscScalar*, DefaultMemorySpace> v("v", n);

    PetscFunctionBeginUser;
    // Simulation of GPU based finite assembly process with COO
    Kokkos::parallel_for(
        "AssembleElementMatrices", n, KOKKOS_LAMBDA(PetscInt i) { v[i] = float(i) + 1.0; }
    );

    VecSetValuesCOO(b, v.data(), ADD_VALUES);
    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    Vec x, b, u; /* approx solution, RHS, exact solution */
    Mat A;       /* linear system matrix */
    KSP ksp;     /* linear solver context */
    PetscInt n = 10;
    PetscErrorCode ierr;

    ierr = PetscInitialize(&argc, &argv, 0, help);
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);

    PetscInt colIdx[n];
    PetscInt rowIdx[n];

    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, n, n, PETSC_DECIDE, PETSC_DECIDE);
    MatSetType(A, MATAIJKOKKOS);

    VecCreate(PETSC_COMM_SELF, &b);
    VecSetSizes(b, PETSC_DECIDE, n);
    VecSetType(b, VECKOKKOS);

    for (int i = 0; i < n; i++)
    {
        colIdx[i] = i;
        rowIdx[i] = i;
    }

    printf("hallo\n");
    MatSetPreallocationCOO(A, n, colIdx, rowIdx);
    VecDuplicate(b, &x);
    VecSetPreallocationCOO(b, n, rowIdx);
    // PetscFree2(oor, ooc);

    printf("hallo2\n");
    FillMatrixKokkosCOO(n, A);
    FillVectorKokkosCOO(n, b);
    // MatView(A, PETSC_VIEWER_STDOUT_WORLD);

    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    // KSPSetTolerances(ksp, 1.e-9, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetFromOptions(ksp);
    // KSPSetUp(ksp);


    printf("before\n");
    KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
    ierr = KSPSolve(ksp, b, x);
    printf("after\n");
    KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
    if (n < 15)
    {
        VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    }
    Kokkos::finalize();

    return 0;
}
