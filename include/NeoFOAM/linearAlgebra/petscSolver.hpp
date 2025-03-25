// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#if NF_WITH_PETSC

#include <Kokkos_Core.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat.h>
#include <petscksp.h>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/linearAlgebra/utilities.hpp"
#include "NeoFOAM/linearAlgebra/petscSolverContext.hpp"


namespace NeoFOAM::la::petscSolver
{

template<typename ValueType>
class petscSolver
{

private:

    Executor exec_;
    Dictionary solverDict_;
    Mat Amat_;
    KSP ksp_;

    Vec sol_, rhs_;


public:

    petscSolver(Executor exec, Dictionary solverDict)
        : exec_(exec), solverDict_(solverDict), Amat_(nullptr), ksp_(nullptr), sol_(nullptr),
          rhs_(nullptr)
    {}

    //- Destructor
    virtual ~petscSolver()
    {
        // MatDestroy(&Amat_);
        KSPDestroy(&ksp_);
        // VecDestroy(&sol_);
        // VecDestroy(&rhs_);
    }

    void solve(LinearSystem<ValueType, int>& sys, Field<ValueType>& x)
    {

        /*
                std::size_t size = sys.matrix().nValues();
                std::size_t nrows = sys.rhs().size();
                PetscInt colIdx[size];
                PetscInt rowIdx[size];
                PetscInt rhsIdx[nrows];

                // colIdx = sys.matrix().colIdxs().data();

                auto rowPtrHost = sys.matrix().rowPtrToHost();
                auto colIdxHost = sys.matrix().colIdxsToHost();
                auto rhsHost = sys.rhs().copyToHost();

                for (size_t index = 0; index < nrows; ++index)
                {
                    rhsIdx[index] = static_cast<PetscInt>(index);
                }
                // copy colidx
                // TODO: (this should be done only once when the matrix
                //  topology changes
                for (size_t index = 0; index < size; ++index)
                {
                    colIdx[index] = static_cast<PetscInt>(colIdxHost[index]);
                }
                // convert rowPtr to rowIdx
                // TODO: (this should be done only once when the matrix
                //  topology changes
                size_t rowI = 0;
                size_t rowOffset = rowPtrHost[rowI + 1];
                for (size_t index = 0; index < size; ++index)
                {
                    if (index == rowOffset)
                    {
                        rowI++;
                        rowOffset = rowPtrHost[rowI + 1];
                    }
                    rowIdx[index] = rowI;
                }

                // move all not necessary staff to outer most scope since matrix  has
                // to be preallocated only once every time the mesh changes
                PetscInitialize(NULL, NULL, 0, NULL);

                MatCreate(PETSC_COMM_WORLD, &Amat_);
                MatSetSizes(Amat_, sys.matrix().nRows(), sys.rhs().size(), PETSC_DECIDE,
           PETSC_DECIDE);

                VecCreate(PETSC_COMM_SELF, &rhs_);
                VecSetSizes(rhs_, PETSC_DECIDE, nrows);

                std::string execName = std::visit([](const auto& e) { return e.name(); }, exec_);

                if (execName == "GPUExecutor")
                {
                    VecSetType(rhs_, VECKOKKOS);
                    MatSetType(Amat_, MATAIJKOKKOS);
                }
                else
                {
                    VecSetType(rhs_, VECSEQ);
                    MatSetType(Amat_, MATSEQAIJ);
                }
                VecDuplicate(rhs_, &sol_);

                VecSetPreallocationCOO(rhs_, nrows, rhsIdx);
                MatSetPreallocationCOO(Amat_, size, colIdx, rowIdx);

                VecSetValuesCOO(rhs_, sys.rhs().data(), ADD_VALUES);
                MatSetValuesCOO(Amat_, sys.matrix().values().data(), ADD_VALUES);
        */
        std::size_t nrows = sys.rhs().size();
        NeoFOAM::la::petscSolverContext::petscSolverContext<ValueType> petsctx(exec_);
        petsctx.initialize(sys);
        Amat_ = petsctx.AMat();
        rhs_ = petsctx.rhs();
        sol_ = petsctx.sol();

        VecSetValuesCOO(rhs_, sys.rhs().data(), ADD_VALUES);
        MatSetValuesCOO(Amat_, sys.matrix().values().data(), ADD_VALUES);

        // MatView(Amat_, PETSC_VIEWER_STDOUT_WORLD);
        // VecView(rhs_, PETSC_VIEWER_STDOUT_WORLD);


        KSPCreate(PETSC_COMM_WORLD, &ksp_);
        KSPSetOperators(ksp_, Amat_, Amat_);
        // KSPSetTolerances(ksp, 1.e-9, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
        KSPSetFromOptions(ksp_);
        // KSPSetUp(ksp);


        PetscCallVoid(KSPSolve(ksp_, rhs_, sol_));


        PetscScalar* x_help = static_cast<PetscScalar*>(x.data());
        VecGetArray(sol_, &x_help);

        NeoFOAM::Field<NeoFOAM::scalar> x2(
            x.exec(), static_cast<ValueType*>(x_help), nrows, x.exec()
        );
        x = x2;


        // PetscCall(ierr);
    }
};

}

#endif
