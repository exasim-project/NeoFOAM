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

    NeoFOAM::Database db_;


public:

    petscSolver(Executor exec, Dictionary solverDict, NeoFOAM::Database db)
        : exec_(exec), solverDict_(solverDict), Amat_(nullptr), ksp_(nullptr), sol_(nullptr),
          rhs_(nullptr), db_(db)
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
