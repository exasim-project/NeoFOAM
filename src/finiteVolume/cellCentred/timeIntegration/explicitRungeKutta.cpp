// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/explicitRungeKutta.hpp"
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"


#include <arkode/arkode_erkstep.h>

namespace NeoFOAM::finiteVolume::cellCentred
{

int explicitSolveWrapperFreeFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
    // Cast user_data to ExplicitRungeKutta* to access the instance
    // ExplicitRungeKutta* self = static_cast<ExplicitRungeKutta*>(user_data);
    std::cout << "\nHello from explicitSolveWrapperFreeFunction\n";
    // Call the non-static member function explicitSolve on the instance
    return 1; // self->explicitSolve(t, y, ydot, user_data);
}

ExplicitRungeKutta::ExplicitRungeKutta(dsl::EqnSystem eqnSystem, const Dictionary& dict)
    : TimeIntegrationFactory::Register<ExplicitRungeKutta>(eqnSystem, dict)
{
    initNDData();
    initSUNContext();
    initSUNARKODESolver();
}

ExplicitRungeKutta::ExplicitRungeKutta(const ExplicitRungeKutta& other)
    : TimeIntegrationFactory::Register<ExplicitRungeKutta>(other),
      data_(
          other.data_ ? std::make_unique<NFData>(*other.data_) : nullptr
      ) // Deep copy of unique_ptr
{
    solution_ = other.solution_;
    context_ = other.context_;
}

void ExplicitRungeKutta::solve()
{
    while (time_ < data_->endTime_)
    {
        time_ += data_->fixedStepSize_;

        std::cout << "Step t = " << time_ << std::endl;
    }
}

std::unique_ptr<TimeIntegrationFactory> ExplicitRungeKutta::clone() const
{
    return std::make_unique<ExplicitRungeKutta>(*this);
}

void ExplicitRungeKutta::initNDData()
{
    data_ = std::make_unique<NFData>();
    data_->realTol_ = dict_.get<scalar>("Relative Tolerance");
    data_->absTol_ = dict_.get<scalar>("Absolute Tolerance");
    data_->fixedStepSize_ = dict_.get<scalar>("Fixed Step Size"); // zero for adaptive
    data_->endTime_ = dict_.get<scalar>("End Time");

    data_->System = dsl::EqnSystem(eqnSystem_);

    data_->nodes = 1;
    data_->maxsteps = 1;
};


void ExplicitRungeKutta::initSUNContext()
{
    int flag = SUNContext_Create(SUN_COMM_NULL, &context_);
    NF_ASSERT(flag == 0, "SUNContext_Create failed");
}

void ExplicitRungeKutta::initSUNARKODESolver()
{
    // kokkosSolution_ = VecType(data_->nodes, context_);
    // solution_ = kokkosSolution_;

    // this->explicitSolve();
    arkodeMemory_.reset(reinterpret_cast<char*>(
        ERKStepCreate(explicitSolveWrapperFreeFunction, 0.0, solution_, context_)
    ));
    void* ark = reinterpret_cast<void*>(arkodeMemory_.get());

    // Initialize ERKStep solver
    ERKStepSetUserData(ark, NULL);
    ERKStepSetInitStep(ark, data_->fixedStepSize_);
    ERKStepSetFixedStep(ark, data_->fixedStepSize_);

    ERKStepSetTableNum(arkodeMemory_.get(), ARKODE_HEUN_EULER_2_1_2);

    ARKStepSStolerances(arkodeMemory_.get(), data_->realTol_, data_->absTol_);
}

int ExplicitRungeKutta::explicitSolve(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
    // int flag =
    //     ARKStepEvolve(arkodeMemory_.get(), time_ + timeStepSize_, solution_, &time_,
    //     ARK_ONE_STEP);

    // if (flag < 0)
    // {
    //     std::cout << "Integration failed with flag: " << flag << std::endl;
    // }
}


} // namespace NeoFOAM
