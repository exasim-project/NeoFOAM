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
    NFData* nfData = reinterpret_cast<NFData*>(user_data);
    Field<scalar> source = nfData->system_.explicitOperation();

    // for (auto& eqnTerm : eqnSystem_.temporalTerms())
    // {
    //     eqnTerm.temporalOperation(Phi);
    // }
    // Phi += source*dt;
    // refField->internalField() -= source * dt;
    // refField->correctBoundaryConditions();

    // check if execturo is GPU
    // if (std::holds_alternative<NeoFOAM::GPUExecutor>(nfData->system_.exec()))
    // {
    //     Kokkos::fence();
    // }
    std::cout << "\n--" << nfData->nodes << "\n";
    std::cout << "\n--" << N_VGetLength(y) << "\n";
    std::cout << "\n--" << N_VGetLength(ydot) << "\n";

    sunrealtype* ydotarray = N_VGetArrayPointer(ydot);
    sunrealtype* yarray = N_VGetArrayPointer(y);

    if (ydotarray == NULL)
    {
        return -1;
    }
    if (yarray == NULL)
    {
        return -1;
    }

    for (std::size_t i = 0; i < nfData->nodes; ++i)
    {
        ydotarray[i] = -1.0 * yarray[i];
    }

    // some kind of memory leak below - need to fix. Is y and ydot sized correctly?
    // NV_Ith_S(ydot, 0) = -1.0 * NV_Ith_S(y, 0);
    return 0; // set 0 -> success
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
        ARKStepEvolve(
            arkodeMemory_.get(), time_ + data_->fixedStepSize_, solution_, &time_, ARK_ONE_STEP
        );
        std::cout << "Step t = " << time_ << "\t" << NV_Ith_S(solution_, 0) << std::endl;
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

    data_->system_ = dsl::EqnSystem(eqnSystem_);

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
    // NOTE CHECK https://sundials.readthedocs.io/en/latest/arkode/Usage/Skeleton.html for order of
    // initialization.


    kokkosSolution_ = VecType(data_->nodes, context_);
    kokkosInitialConditions_ = VecType(data_->nodes, context_);
    solution_ = kokkosSolution_;
    initialConditions_ = kokkosInitialConditions_;

    N_VConst(1.0, initialConditions_);
    arkodeMemory_.reset(reinterpret_cast<char*>(
        ERKStepCreate(explicitSolveWrapperFreeFunction, 0.0, initialConditions_, context_)
    ));
    void* ark = reinterpret_cast<void*>(arkodeMemory_.get());

    // Initialize ERKStep solver
    ERKStepSetUserData(ark, NULL);
    ERKStepSetInitStep(ark, data_->fixedStepSize_);
    ERKStepSetFixedStep(ark, data_->fixedStepSize_);

    ERKStepSetTableNum(arkodeMemory_.get(), ARKODE_FORWARD_EULER_1_1);

    ARKStepSStolerances(arkodeMemory_.get(), data_->realTol_, data_->absTol_);
    ARKodeSetUserData(arkodeMemory_.get(), data_.get());
}


} // namespace NeoFOAM
