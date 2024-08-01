// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/explicitRungeKutta.hpp"
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"


#include <arkode/arkode_erkstep.h>

namespace NeoFOAM::finiteVolume::cellCentred
{

ExplicitRungeKutta::ExplicitRungeKutta(const dsl::EqnSystem& eqnSystem, const Dictionary& dict)
    : TimeIntegrationFactory::Register<ExplicitRungeKutta>(eqnSystem, dict)
{

    // NF_ASSERT(system_.exec() == Executor(CPUExecutor), "SundialsIntergrator currently only
    // supports CPU execution");
    initNDData();
    initSUNContext();
    initSUNARKODESolver();
    initSUNLinearSolver();
}

ExplicitRungeKutta::ExplicitRungeKutta(const ExplicitRungeKutta& other)
    : Base(other), // Call the base class copy constructor
      data_(
          other.data_ ? std::make_unique<NFData>(*other.data_) : nullptr
      ) // Deep copy of unique_ptr
{

    solution_ = other.solution;
    context_ = other.context_;
    linearSolver_ = other.linearSolver_;
}

void ExplicitRungeKutta::solve()
{
    ARKStepEvolve(
        reinterpret_cast<void*>(arkodeMemory_.get()), 0.0, solution_, nullptr, ARK_ONE_STEP
    );
}

std::unique_ptr<TimeIntegrationFactory> ExplicitRungeKutta::clone() const
{
    return std::make_unique<ExplicitRungeKutta>(*this);
}

void ExplicitRungeKutta::initNDData()
{
    data_ = std::make_unique<NFData>();

    data_->forcing = true; // forces error output
    data_->realTol_ = dict.get<scalar>("Relative Tolerance");
    data_->absTol_ = dict.get<scalar>("Absolute Tolerance");
    data_->fixedStepSize_ = dict.get<scalar>("Fixed Step Size"); // zero for adaptive
    data_->order = 1;                                            // Temporal order of the method
    data_->controller =
        ARKAdaptControllerType::PID; // dummy value, currently the timestep size is fixed .
    data_->maxsteps = 1;
    data_->linear = false;
    data_->diagnostics = true;

    data_->pcg = true;          // use PCG (true) or GMRES (false)
    data_->precondition = true; // enable preconditioning
    data_->lsinfo = false;      // output residual history
    data_->liniters = 40;       // max linear iterations
    data_->msbp = 20;           // use default (20 steps)
    data_->epslin = 0.05;       // use default (0.05)


    // Output variables
    data_->output = 1; // 0 = no output, 1 = stats output, 2 = output to disk
    data_->nout = 20;  // Number of output times
    data_->e = nullptr;

    // Timing variables
    data_->timing = false;
    data_->evolvetime = 0.0;
    data_->rhstime = 0.0;
    data_->psetuptime = 0.0;
    data_->psolvetime = 0.0;

    data_->nodes = 1;
};


void ExplicitRungeKutta::initSUNContext()
{
    int flag = SUNContext_Create(SUN_COMM_NULL, &context_);
    NF_ASSERT(flag == 0, "SUNContext_Create failed");
}

void ExplicitRungeKutta::initSUNARKODESolver()
{
    kokkosSolution_ = VecType(data_->nodes, context_);
    solution_ = kokkosSolution_;
    void* ark = reinterpret_cast<void*>(arkodeMemory_.get());
    ark = ERKStepCreate(this->solveExplicit(), nullptr, 0.0, solution_, context_);

    ERKStepSetTableNum(arkodeMemory_.get(), ARKODE_HEUN_EULER_2_1_2);

    ARKStepSStolerances(arkodeMemory_.get(), data_->realTol_, data_->absTol_);
}

void ExplicitRungeKutta::explicitSolve()
{
    // std::cout << "Solving using Forward Euler" << std::endl;w
    // scalar dt = 0.001; // Time step
    // fvcc::VolumeField<scalar>* refField = eqnSystem_.volumeField();
    // // Field<scalar> Phi(eqnSystem_.exec(), eqnSystem_.nCells());
    // // NeoFOAM::fill(Phi, 0.0);
    // Field<scalar> source = eqnSystem_.explicitOperation();

    // // for (auto& eqnTerm : eqnSystem_.temporalTerms())
    // // {
    // //     eqnTerm.temporalOperation(Phi);
    // // }
    // // Phi += source*dt;
    // refField->internalField() -= source * dt;
    // refField->correctBoundaryConditions();

    // // check if execturo is GPU
    // if (std::holds_alternative<NeoFOAM::GPUExecutor>(eqnSystem_.exec()))
    // {
    //     Kokkos::fence();
    // }
}


} // namespace NeoFOAM
