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
    data_->realTol_ = dict_.get<scalar>("Relative Tolerance");
    data_->absTol_ = dict_.get<scalar>("Absolute Tolerance");
    data_->fixedStepSize_ = dict_.get<scalar>("Fixed Step Size"); // zero for adaptive
    data_->maxsteps = 1;


    // Output variables
    data_->output = 1; // 0 = no output, 1 = stats output, 2 = output to disk
    data_->nout = 20;  // Number of output times

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
    // kokkosSolution_ = VecType(data_->nodes, context_);
    // solution_ = kokkosSolution_;
    void* ark = reinterpret_cast<void*>(arkodeMemory_.get());
    // this->explicitSolve();
    ark = ERKStepCreate(nullptr, 0.0, solution_, context_);

    ERKStepSetTableNum(arkodeMemory_.get(), ARKODE_HEUN_EULER_2_1_2);

    ARKStepSStolerances(arkodeMemory_.get(), data_->realTol_, data_->absTol_);
}

void ExplicitRungeKutta::explicitSolve()
{
    int flag =
        ARKStepEvolve(arkodeMemory_.get(), time_ + timeStepSize_, solution_, &time_, ARK_ONE_STEP);

    if (flag < 0)
    {
        std::cout << "Integration failed with flag: " << flag << std::endl;
    }

    std::cout << "Step t = " << time_ << std::endl;
}


} // namespace NeoFOAM
