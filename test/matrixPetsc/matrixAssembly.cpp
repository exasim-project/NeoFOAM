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


TEST_CASE("matrix assembly")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("Matrix assembly" + execName)
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

        Mat A;

        PetscInitialize(NULL, NULL, NULL, NULL);

        MatCreate(PETSC_COMM_WORLD, &A);
        MatSetSizes(A, size, size, PETSC_DECIDE, PETSC_DECIDE);
        if (execName == "GPUExecutor")
        {
            MatSetType(A, MATAIJKOKKOS);
        }
        else
        {
            MatSetType(A, MATSEQAIJ);
        }
        MatSetPreallocationCOO(A, size, colIdx, rowIdx);
        // PetscFree2(oor, ooc);

        MatSetValuesCOO(A, values.data(), ADD_VALUES);

        // MatView(A, PETSC_VIEWER_STDOUT_WORLD);
        // MatGetValues(A, size, rowIdx, 1, colIdx, v);
        for (int i = 1; i < size; i++)
        {
            PetscInt idr[1];
            PetscInt idc[1];

            idr[0] = rowIdx[i];
            idc[0] = colIdx[i];

            MatGetValues(A, 1, idr, 1, idc, w);
            v[i] = w[0];
        }
        // PetscScalarView(size, v, PETSC_VIEWER_STDOUT_WORLD);

        REQUIRE(v[0] == 1.);
        REQUIRE(v[1] == 2.);
        REQUIRE(v[2] == 3.);
        REQUIRE(v[3] == 4.);
        REQUIRE(v[4] == 5.);
        REQUIRE(v[5] == 6.);
        REQUIRE(v[6] == 7.);
        REQUIRE(v[7] == 8.);
        REQUIRE(v[8] == 9.);
        REQUIRE(v[9] == 10.);

        MatDestroy(&A);
    }

    SECTION("rhs assembly" + execName)
    {

        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> values(
            exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
        );
        PetscInt rowIdx[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        PetscScalar v[10];

        Vec b;

        PetscInitialize(NULL, NULL, NULL, NULL);

        VecCreate(PETSC_COMM_SELF, &b);
        VecSetSizes(b, PETSC_DECIDE, size);

        if (execName == "GPUExecutor")
        {
            VecSetType(b, VECKOKKOS);
        }
        else
        {
            VecSetType(b, VECSEQ);
        }

        VecSetPreallocationCOO(b, size, rowIdx);
        VecSetValuesCOO(b, values.data(), ADD_VALUES);

        VecGetValues(b, size, rowIdx, v);

        REQUIRE(v[0] == 1.);
        REQUIRE(v[1] == 2.);
        REQUIRE(v[2] == 3.);
        REQUIRE(v[3] == 4.);
        REQUIRE(v[4] == 5.);
        REQUIRE(v[5] == 6.);
        REQUIRE(v[6] == 7.);
        REQUIRE(v[7] == 8.);
        REQUIRE(v[8] == 9.);
        REQUIRE(v[9] == 10.);

        VecDestroy(&b);
    }

    SECTION("vector matrix multiplication" + execName)
    {

        NeoFOAM::size_t size = 10;
        NeoFOAM::Field<NeoFOAM::scalar> values(
            exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
        );
        PetscInt rowIdx[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        PetscInt colIdx[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        PetscScalar v[10];

        Vec b, u;
        Mat A;

        PetscInitialize(NULL, NULL, NULL, NULL);

        VecCreate(PETSC_COMM_SELF, &b);
        VecSetSizes(b, PETSC_DECIDE, size);
        MatCreate(PETSC_COMM_WORLD, &A);
        MatSetSizes(A, size, size, PETSC_DECIDE, PETSC_DECIDE);

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
        VecDuplicate(b, &u);

        VecSetPreallocationCOO(b, size, rowIdx);
        VecSetValuesCOO(b, values.data(), ADD_VALUES);

        MatSetPreallocationCOO(A, size, colIdx, rowIdx);
        MatSetValuesCOO(A, values.data(), ADD_VALUES);

        MatMult(A, b, u);
        VecGetValues(u, size, rowIdx, v);

        REQUIRE(v[0] == 1.);
        REQUIRE(v[1] == 4.);
        REQUIRE(v[2] == 9.);
        REQUIRE(v[3] == 16.);
        REQUIRE(v[4] == 25.);
        REQUIRE(v[5] == 36.);
        REQUIRE(v[6] == 49.);
        REQUIRE(v[7] == 64.);
        REQUIRE(v[8] == 81.);
        REQUIRE(v[9] == 100.);

        MatDestroy(&A);
        VecDestroy(&u);
        VecDestroy(&b);
    }
}
