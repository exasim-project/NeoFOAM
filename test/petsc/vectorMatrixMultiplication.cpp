// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors


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
#include <petscksp.h>
#include <petscsys.h>
#include <petscviewer.h>

int main(int argc, char** args)
{
    Vec x, b, u;    /* approx solution, RHS, exact solution */
    Mat A;          /* linear system matrix */
    KSP ksp;        /* linear solver context */
    PC pc;          /* preconditioner context */
    PetscReal norm; /* norm of solution error */
    PetscInt i, n = 10, col[3], its;
    PetscMPIInt size;
    PetscScalar value[3];
    PetscBool has;
    PetscOptions options;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &args, NULL, help));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(
        size == 1,
        PETSC_COMM_WORLD,
        PETSC_ERR_WRONG_MPI_SIZE,
        "This is a uniprocessor example only!"
    );

    // PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
    PetscCall(PetscOptionsSetValue(NULL, "-use_gpu_aware_mpi", "0"));
    PetscCall(PetscOptionsHasName(NULL, NULL, "-use_gpu_aware_mpi", &has));
    PetscTestCheck(has == PETSC_TRUE);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute the matrix and right-hand-side vector that define
           the linear system, Ax = b.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Create vectors.  Note that we form 1 vector from scratch and
       then duplicate as needed.
    */
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Hallo"));
    PetscCall(VecCreate(PETSC_COMM_SELF, &x));
    PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
    // PetscCall(VecSetFromOptions(x));
    // PetscCall(VecSetType(x, VECSEQ));
    PetscCall(VecSetType(x, VECCUDA));
    PetscCall(VecDuplicate(x, &b));
    PetscCall(VecDuplicate(x, &u));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Hallo"));


    /*
       Create matrix.  When using MatCreate(), the matrix format can
       be specified at runtime.

       Performance tuning note:  For problems of substantial size,
       preallocation of matrix memory is crucial for attaining good
       performance. See the matrix chapter of the users manual for details.
    */
    PetscCall(MatCreate(PETSC_COMM_SELF, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
    // PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetType(A, MATSEQAIJCUSPARSE));
    // PetscCall(MatSetType(A, MATSEQBAIJ));
    PetscCall(MatSetUp(A));

    /*
       Assemble matrix
    */
    value[0] = -1.0;
    value[1] = 2.0;
    value[2] = -1.0;
    for (i = 1; i < n - 1; i++)
    {
        col[0] = i - 1;
        col[1] = i;
        col[2] = i + 1;
        PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
    }
    if (n > 1)
    {
        i = n - 1;
        col[0] = n - 2;
        col[1] = n - 1;
        PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
    }
    i = 0;
    col[0] = 0;
    col[1] = 1;
    value[0] = 2.0;
    value[1] = -1.0;
    PetscCall(MatSetValues(A, 1, &i, n > 1 ? 2 : 1, col, value, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    /*
       Set exact solution; then compute right-hand-side vector.
    */
    PetscCall(VecSet(u, 1.0));
    PetscCall(MatMult(A, u, b));


    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));

    /*
       Always call PetscFinalize() before exiting a program.  This routine
         - finalizes the PETSc libraries as well as MPI
         - provides summary and diagnostic information if certain runtime
           options are chosen (e.g., -log_view).
    */
    PetscCall(PetscFinalize());
    return 0;
} // SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors


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
#include <petscksp.h>
#include <petscsys.h>
#include <petscviewer.h>

int main(int argc, char** args)
{
    Vec x, b, u;    /* approx solution, RHS, exact solution */
    Mat A;          /* linear system matrix */
    KSP ksp;        /* linear solver context */
    PC pc;          /* preconditioner context */
    PetscReal norm; /* norm of solution error */
    PetscInt i, n = 10, col[3], its;
    PetscMPIInt size;
    PetscScalar value[3];
    PetscBool has;
    PetscOptions options;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &args, NULL, help));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(
        size == 1,
        PETSC_COMM_WORLD,
        PETSC_ERR_WRONG_MPI_SIZE,
        "This is a uniprocessor example only!"
    );

    // PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
    PetscCall(PetscOptionsSetValue(NULL, "-use_gpu_aware_mpi", "0"));
    PetscCall(PetscOptionsHasName(NULL, NULL, "-use_gpu_aware_mpi", &has));
    PetscTestCheck(has == PETSC_TRUE);
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute the matrix and right-hand-side vector that define
           the linear system, Ax = b.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Create vectors.  Note that we form 1 vector from scratch and
       then duplicate as needed.
    */
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Hallo"));
    PetscCall(VecCreate(PETSC_COMM_SELF, &x));
    PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
    // PetscCall(VecSetFromOptions(x));
    // PetscCall(VecSetType(x, VECSEQ));
    PetscCall(VecSetType(x, VECCUDA));
    PetscCall(VecDuplicate(x, &b));
    PetscCall(VecDuplicate(x, &u));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Hallo"));


    /*
       Create matrix.  When using MatCreate(), the matrix format can
       be specified at runtime.

       Performance tuning note:  For problems of substantial size,
       preallocation of matrix memory is crucial for attaining good
       performance. See the matrix chapter of the users manual for details.
    */
    PetscCall(MatCreate(PETSC_COMM_SELF, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
    // PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetType(A, MATSEQAIJCUSPARSE));
    // PetscCall(MatSetType(A, MATSEQBAIJ));
    PetscCall(MatSetUp(A));

    /*
       Assemble matrix
    */
    value[0] = -1.0;
    value[1] = 2.0;
    value[2] = -1.0;
    for (i = 1; i < n - 1; i++)
    {
        col[0] = i - 1;
        col[1] = i;
        col[2] = i + 1;
        PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
    }
    if (n > 1)
    {
        i = n - 1;
        col[0] = n - 2;
        col[1] = n - 1;
        PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
    }
    i = 0;
    col[0] = 0;
    col[1] = 1;
    value[0] = 2.0;
    value[1] = -1.0;
    PetscCall(MatSetValues(A, 1, &i, n > 1 ? 2 : 1, col, value, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    /*
       Set exact solution; then compute right-hand-side vector.
    */
    PetscCall(VecSet(u, 1.0));
    PetscCall(MatMult(A, u, b));


    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));

    /*
       Always call PetscFinalize() before exiting a program.  This routine
         - finalizes the PETSc libraries as well as MPI
         - provides summary and diagnostic information if certain runtime
           options are chosen (e.g., -log_view).
    */
    PetscCall(PetscFinalize());
    return 0;
}
