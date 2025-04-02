// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
// inspired by
// https://develop.openfoam.com/modules/external-solver/-/blob/develop/src/petsc4Foam/utils/petscLinearSolverContext.H

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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace NeoFOAM::la::petscSolverContext
{

template<typename ValueType>
class petscSolverContext
{
    // Private Data

    bool init_, updated_;
    Executor exec_;
    Mat Amat_;
    // KSP ksp_;

    Vec sol_, rhs_;

public:


    // Constructors

    //- Default construct
    petscSolverContext(Executor exec)
        : init_(false), updated_(false), exec_(exec), Amat_(nullptr), sol_(nullptr),
          rhs_(nullptr) //, ksp_(nullptr)
    {}


    //- Destructor
    virtual ~petscSolverContext()
    {
        MatDestroy(&Amat_);
        // KSPDestroy(&ksp_);
        VecDestroy(&sol_);
        VecDestroy(&rhs_);
    }


    // Member Functions

    //- Return value of initialized
    bool initialized() const noexcept { return init_; }


    //- Return value of initialized
    bool updated() const noexcept { return updated_; }

    //- Create auxiliary rows for calculation purposes
    void initialize(LinearSystem<ValueType, int>& sys)
    {
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
        MatSetSizes(Amat_, sys.matrix().nRows(), sys.rhs().size(), PETSC_DECIDE, PETSC_DECIDE);

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

        init_ = true;
    }

    //- Create auxiliary rows for calculation purposes
    void update() { NF_ERROR_EXIT("Mesh changes not supported"); }

    [[nodiscard]] Mat& AMat() { return Amat_; }

    [[nodiscard]] Vec& rhs() { return rhs_; }

    [[nodiscard]] Vec& sol() { return sol_; }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End NeoFOAM::la::petscSolverContext

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
