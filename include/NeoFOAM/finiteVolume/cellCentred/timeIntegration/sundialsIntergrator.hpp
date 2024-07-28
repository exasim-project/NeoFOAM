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
#include <sundials/sundials_core.hpp>
#include <sunlinsol/sunlinsol_kokkosdense.hpp>
#include <sunmatrix/sunmatrix_kokkosdense.hpp>

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

    // Enable/disable forcing (what does this do?)
    bool forcing;

    // Final time
    sunrealtype tf;

    // Integrator settings
    sunrealtype realTol_;                        // relative tolerance
    sunrealtype absTol_;                         // absolute tolerance
    sunrealtype fixedStepSize_;                  // fixed step size
    int order;                                   // ARKode method order
    sundials::ARKAdaptControllerType controller; // step size adaptivity method (0 == fixedStepSize_
                                                 // -> fixed step size controller number ignored)
    int maxsteps;                                // max number of steps between outputs
    bool linear;                                 // enable/disable linearly implicit option
    bool diagnostics;                            // output diagnostics

    // Linear solver and preconditioner settings
    bool pcg;           // use PCG (true) or GMRES (false)
    bool precondition;  // use a preconditioner (true - we will use one)
    bool lsinfo;        // output residual history
    int liniters;       // number of linear iterations
    int msbp;           // max number of steps between preconditioner setups
    sunrealtype epslin; // linear solver tolerance factor

    // Inverse of Jacobian diagonal for preconditioner
    std::unique_ptr<N_Vector> diagonal;

    // Output variables
    int output; // output level
    int nout;   // number of output times
    // ofstream uout; // output file stream
    // ofstream eout; // error file stream
    N_Vector e; // error vector

    // Timing variables
    bool timing; // print timings
    double evolvetime;
    double rhstime;
    double psetuptime;
    double psolvetime;
};


class SundialsIntergrator : public TimeIntegrationFactory::Register<SundialsIntergrator>
{
    using Base = TimeIntegrationFactory::Register<SundialsIntergrator>;
    using VecType = sundials::kokkos::Vector<ExecSpace>;
    using MatType = sundials::kokkos::DenseMatrix<ExecSpace>;
    using LSType = sundials::kokkos::DenseLinearSolver<ExecSpace>;
    using SizeType = VecType::size_type;

public:

    SundialsIntergrator() = default;
    ~SundialsIntergrator() = default;

    SundialsIntergrator(const SundialsIntergrator& other);

    inline SundialsIntergrator& operator=(const SundialsIntergrator& other)
    {
        *this = SundialsIntergrator(other);
        return *this;
    };

    SundialsIntergrator(const dsl::EqnSystem& eqnSystem, const Dictionary& dict);

    static std::string name() { return "SundialsIntergrator"; }

    static std::string doc() { return "SundialsIntergrator timeIntegration"; }

    static std::string schema() { return "none"; }

    void solve() override;

    std::unique_ptr<TimeIntegrationFactory> clone() const override;


private:

    N_Vector solution_;

    SUNContext context_;
    SUNLinearSolver linearSolver_;
    SUNAdaptController controller_;
    std::unique_ptr<char> arkodeMemory_; // this should be void* but that is not stl compliant we
                                         // store the next best thing.
    std::unique_ptr<NFData> data_;


    void initNDData();
    void initSUNContext();
    void initSUNLinearSolver();
    void initSUNAdaptController(); // I don't think we need a time controller.
};

} // namespace NeoFOAM
