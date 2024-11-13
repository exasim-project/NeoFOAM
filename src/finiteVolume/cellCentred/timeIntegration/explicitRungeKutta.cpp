// // SPDX-License-Identifier: MIT
// // SPDX-FileCopyrightText: 2023 NeoFOAM authors

// #include <memory>

// #include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/explicitRungeKutta.hpp"
// #include "NeoFOAM/core/error.hpp"
// #include "NeoFOAM/core/parallelAlgorithms.hpp"
#include <memory>

#include "NeoFOAM/dsl/timeIntegration/explicitRungeKutta.hpp"
#include "NeoFOAM/core/error.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{
} // namespace NeoFOAM


// #include <arkode/arkode_erkstep.h>

// namespace NeoFOAM::dsl
// {

// int explicitSolveWrapperFreeFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
// {
//     // Pointer wrangling
//     NFData* nfData = reinterpret_cast<NFData*>(user_data);
//     sunrealtype* ydotarray = N_VGetArrayPointer(ydot);
//     sunrealtype* yarray = N_VGetArrayPointer(y);

//     if (ydotarray == NULL)
//     {
//         return -1;
//     }
//     if (yarray == NULL)
//     {
//         return -1;
//     }

//     // TODO: copy the field to the solution

//     // solve the spacil terms
//     // Field<scalar> source = nfData->system_.explicitOperation();

//     for (std::size_t i = 0; i < nfData->nodes; ++i)
//     {
//         ydotarray[i] = -1.0 * yarray[i]; // replace with source[i]
//     }

//     // for (auto& eqnTerm : eqnSystem_.temporalTerms())
//     // {
//     //     eqnTerm.temporalOperation(Phi);
//     // }
//     // Phi += source*dt;
//     // refField->internalField() -= source * dt;
//     // refField->correctBoundaryConditions();

//     // check if execturo is GPU
//     if (std::holds_alternative<NeoFOAM::GPUExecutor>(nfData->system_.exec()))
//     {
//         Kokkos::fence();
//     }
//     return 0;
// }


// ExplicitRungeKutta::ExplicitRungeKutta(dsl::EqnSystem eqnSystem, const Dictionary& dict)
//     : TimeIntegrationFactory::Register<ExplicitRungeKutta>(eqnSystem, dict)
// {
//     initSUNERKSolver();
// }

// ExplicitRungeKutta::ExplicitRungeKutta(const ExplicitRungeKutta& other)
//     : TimeIntegrationFactory::Register<ExplicitRungeKutta>(other),
//       data_(
//           other.data_ ? std::make_unique<NFData>(*other.data_) : nullptr
//       ) // Deep copy of unique_ptr
// {
//     solution_ = other.solution_;
//     context_ = other.context_;
// }

// void ExplicitRungeKutta::solve()
// {
//     while (time_ < data_->endTime_)
//     {
//         time_ += data_->fixedStepSize_;
//         ARKStepEvolve(
//             arkodeMemory_.get(), time_ + data_->fixedStepSize_, solution_, &time_, ARK_ONE_STEP
//         );
//         sunrealtype* solution = N_VGetArrayPointer(solution_);
//         std::cout << "Step t = " << time_ << "\t" << solution[0] << std::endl;
//     }
// }

// std::unique_ptr<TimeIntegrationFactory> ExplicitRungeKutta::clone() const
// {
//     return std::make_unique<ExplicitRungeKutta>(*this);
// }

// void ExplicitRungeKutta::initSUNERKSolver()
// {
//     // NOTE CHECK https://sundials.readthedocs.io/en/latest/arkode/Usage/Skeleton.html for order
//     of
//     // initialization.
//     initNFData();

//     // Initialize SUNdials solver;
//     initSUNContext();
//     initSUNDimension();
//     initSUNInitialConditions();
//     initSUNCreateERK();
//     initSUNTolerances();
// }

// void ExplicitRungeKutta::initNFData()
// {
//     data_ = std::make_unique<NFData>();
//     data_->realTol_ = dict_.get<scalar>("Relative Tolerance");
//     data_->absTol_ = dict_.get<scalar>("Absolute Tolerance");
//     data_->fixedStepSize_ = dict_.get<scalar>("Fixed Step Size"); // zero for adaptive
//     data_->endTime_ = dict_.get<scalar>("End Time");

//     data_->system_ = dsl::EqnSystem(eqnSystem_);

//     data_->nodes = 1;
//     data_->maxsteps = 1;
// };

// void ExplicitRungeKutta::initSUNContext()
// {
//     int flag = SUNContext_Create(SUN_COMM_NULL, &context_);
//     NF_ASSERT(flag == 0, "SUNContext_Create failed");
// }

// void ExplicitRungeKutta::initSUNDimension()
// {
//     kokkosSolution_ = VecType(data_->nodes, context_);
//     kokkosInitialConditions_ = VecType(data_->nodes, context_);
//     solution_ = kokkosSolution_;
//     initialConditions_ = kokkosInitialConditions_;
// }

// void ExplicitRungeKutta::initSUNInitialConditions() { N_VConst(1.0, initialConditions_); }

// void ExplicitRungeKutta::initSUNCreateERK()
// {
//     arkodeMemory_.reset(reinterpret_cast<char*>(
//         ERKStepCreate(explicitSolveWrapperFreeFunction, 0.0, initialConditions_, context_)
//     ));
//     void* ark = reinterpret_cast<void*>(arkodeMemory_.get());

//     // Initialize ERKStep solver
//     ERKStepSetUserData(ark, NULL);
//     ERKStepSetInitStep(ark, data_->fixedStepSize_);
//     ERKStepSetFixedStep(ark, data_->fixedStepSize_);
//     ERKStepSetTableNum(ark, ARKODE_FORWARD_EULER_1_1);
//     ARKodeSetUserData(ark, data_.get());
// }

// void ExplicitRungeKutta::initSUNTolerances()
// {
//     ARKStepSStolerances(arkodeMemory_.get(), data_->realTol_, data_->absTol_);
// }


// } // namespace NeoFOAM