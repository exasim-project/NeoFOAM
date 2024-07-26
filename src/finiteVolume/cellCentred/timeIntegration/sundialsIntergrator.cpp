// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/sundialsIntergrator.hpp"
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


SundialsIntergrator::SundialsIntergrator(const dsl::EqnSystem& eqnSystem, const Dictionary& dict)
    : TimeIntegrationFactory::Register<SundialsIntergrator>(eqnSystem, dict)
{

    // NF_ASSERT(system_.exec() == Executor(CPUExecutor), "SundialsIntergrator currently only
    // supports CPU execution");
    data_ = std::make_unique<NFData>();

    data_->forcing = true; // forces error output
    data_->realTol_ = dict.get<scalar>("Relative Tolerance");
    data_->absTol_ = dict.get<scalar>("Absolute Tolerance");
    data_->fixedStepSize_ = dict.get<scalar>("Fixed Step Size"); // zero for adaptive
    data_->order = 1;                                            // Temporal order of the method
    data_->controller = 0; // currently the timestep size is fixed .
    data_->maxsteps = 1;
    data_->linear = false;
    data_->diagnostics = true;

    data_->pcg = true;     // use PCG (true) or GMRES (false)
    data_->prec = true;    // enable preconditioning
    data_->lsinfo = false; // output residual history
    data_->liniters = 40;  // max linear iterations
    data_->msbp = 0;       // use default (20 steps)
    data_->epslin = 0.0;   // use default (0.05)

    // Inverse of Jacobian diagonal for preconditioner
    data_->d = nullptr;

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
}

SundialsIntergrator::SundialsIntergrator(const SundialsIntergrator& other)
    : Base(other), // Call the base class copy constructor
      data_(
          other.data_ ? std::make_unique<NFData>(*other.data_) : nullptr
      ) // Deep copy of unique_ptr
{

    solution = other.solution;
    context_ = other.context_;
    linearSolver_ = other.linearSolver_;
}

void SundialsIntergrator::solve() {}

std::unique_ptr<TimeIntegrationFactory> SundialsIntergrator::clone() const
{
    return std::make_unique<SundialsIntergrator>(*this);
}

} // namespace NeoFOAM
