// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/timeIntegration.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

#include <functional>
#include <memory>

// possibly useful headers.
#include <nvector/nvector_kokkos.hpp>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_core.hpp>
// #include <sunlinsol/sunlinsol_kokkosdense.hpp>
// #include <sunmatrix/sunmatrix_kokkosdense.hpp>

#include "sundails.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

#if defined(USE_CUDA)
using ExecSpace = Kokkos::Cuda;
#elif defined(USE_HIP)
#if KOKKOS_VERSION / 10000 > 3
using ExecSpace = Kokkos::HIP;
#else
using ExecSpace = Kokkos::Experimental::HIP;
#endif
#elif defined(USE_OPENMP)
using ExecSpace = Kokkos::OpenMP;
#else
using ExecSpace = Kokkos::Serial;
#endif

// Where to put?
struct NFData
{
    // Final time
    sunrealtype tf;

    // Integrator settings
    sunrealtype realTol_;       // relative tolerance
    sunrealtype absTol_;        // absolute tolerance
    sunrealtype fixedStepSize_; // fixed step size
    sunrealtype endTime_;       // end time
    int order;                  // ARKode method order
                                // -> fixed step size controller number ignored)
    int maxsteps;               // max number of steps between outputs
    int timeStep;               // time step number

    dsl::EqnSystem System;

    // Output variables
    int output; // output level
    int nout;   // number of output times

    // Timing variables
    bool timing; // print timings
    double evolvetime;
    size_t nodes;
};


class ExplicitRungeKutta : public TimeIntegrationFactory::Register<ExplicitRungeKutta>
{
    // using VecType = sundials::kokkos::Vector<ExecSpace>;
    // using SizeType = VecType::size_type;

public:

    ExplicitRungeKutta() = default;
    ~ExplicitRungeKutta() = default;

    ExplicitRungeKutta(const ExplicitRungeKutta& other);

    inline ExplicitRungeKutta& operator=(const ExplicitRungeKutta& other)
    {
        *this = ExplicitRungeKutta(other);
        return *this;
    };

    ExplicitRungeKutta(dsl::EqnSystem eqnSystem, const Dictionary& dict);

    static std::string name() { return "explicitRungeKutta"; }

    static std::string doc() { return "explicitRungeKutta timeIntegration"; }

    static std::string schema() { return "none"; }

    void solve() override;

    std::unique_ptr<TimeIntegrationFactory> clone() const override;

    int explicitSolve(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);

private:

    double timeStepSize_;
    double time_;
    // VecType kokkosSolution_;
    N_Vector solution_;
    SUNContext context_;
    std::unique_ptr<char> arkodeMemory_; // this should be void* but that is not stl compliant we
                                         // store the next best thing.
    std::unique_ptr<NFData> data_;


    void initNDData();
    void initSUNARKODESolver();
    void initSUNContext();
};

} // namespace NeoFOAM
