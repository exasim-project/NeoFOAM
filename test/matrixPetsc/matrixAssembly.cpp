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


TEST_CASE("Field Constructors")
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
        // NeoFOAM::Field<PetscInt> colIdx(exec, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        // NeoFOAM::Field<PetscInt> rowIdx(exec, {0, 1, 2, 4, 4, 5, 6, 7, 8, 9});

        Mat A;

        PetscInitialize(NULL, NULL, NULL, NULL);

        std::cout << values.data() << "\n";

        MatCreate(PETSC_COMM_WORLD, &A);
        MatSetSizes(A, size, size, PETSC_DECIDE, PETSC_DECIDE);
        std::cout << execName << "\n";
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

        std::cout << "after set MatSetValuesC00"
                  << "\n";
        MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    }
}
