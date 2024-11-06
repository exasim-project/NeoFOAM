// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>
#include <memory>

// possibly useful headers.
#include <nvector/nvector_serial.h>
#include <nvector/nvector_kokkos.hpp>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_core.hpp>
// #include <sunlinsol/sunlinsol_kokkosdense.hpp>
// #include <sunmatrix/sunmatrix_kokkosdense.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/timeIntegration/timeIntegration.hpp"
#include "sundials.hpp"


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


struct NFData
{
    NFData() = default;
    ~NFData() = default;

    NFData(const NFData& other)
    {
        // Final time
        tf = other.tf;

        // Integrator settings
        realTol_ = other.realTol_;             // relative tolerance
        absTol_ = other.absTol_;               // absolute tolerance
        fixedStepSize_ = other.fixedStepSize_; // fixed step size
        endTime_ = other.endTime_;             // end time
        order = other.order;                   // ARKode method order
                                               // -> fixed step size controller number ignored)
        maxsteps = other.maxsteps;             // max number of steps between outputs
        timeStep = other.timeStep;             // time step number

        system_ = std::make_unique<NeoFOAM::dsl::Expression>(other.system_->exec()
        ); // system of equations

        // Output variables
        output = other.output; // output level
        nout = other.nout;     // number of output times

        // Timing variables
        timing = other.timing; // print timings
        evolvetime = other.evolvetime;
        nodes = other.nodes;
    }

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

    std::unique_ptr<NeoFOAM::dsl::Expression> system_ {nullptr}; // system of equations

    // Output variables
    int output; // output level
    int nout;   // number of output times

    // Timing variables
    bool timing; // print timings
    double evolvetime;
    size_t nodes;
};

int explicitSolveWrapperFreeFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
    // Pointer wrangling
    NFData* nfData = reinterpret_cast<NFData*>(user_data);
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

    // TODO: copy the field to the solution

    // solve the spacial terms
    NeoFOAM::Field<NeoFOAM::scalar> source(nfData->system_->exec(), 1);
    source = nfData->system_->explicitOperation(source);
    for (std::size_t i = 0; i < nfData->nodes; ++i)
    {
        ydotarray[i] = -1.0 * source[i];
    }

    // refField->correctBoundaryConditions();

    // check if execturo is GPU
    if (std::holds_alternative<NeoFOAM::GPUExecutor>(nfData->system_->exec()))
    {
        Kokkos::fence();
    }
    return 0;
}

// Where to put?
namespace NeoFOAM::dsl
{

template<typename SolutionFieldType>
class ExplicitRungeKutta :
    public TimeIntegratorBase<SolutionFieldType>::template Register<
        ExplicitRungeKutta<SolutionFieldType>>
{
    using VecType = ::sundials::kokkos::Vector<ExecSpace>;
    using SizeType = VecType::size_type;

public:

    using Base = TimeIntegratorBase<SolutionFieldType>::template Register<
        ExplicitRungeKutta<SolutionFieldType>>;
    using Base::dict_;

    ExplicitRungeKutta() = default;

    ~ExplicitRungeKutta() { SUNContext_Free(&context_); };

    ExplicitRungeKutta(const Dictionary& dict) : Base(dict) {}

    ExplicitRungeKutta(const ExplicitRungeKutta& other)
        : Base(other), data_(
                           other.data_ ? std::make_unique<NFData>(*other.data_) : nullptr
                       ) // Deep copy of unique_ptr
    {
        solution_ = other.solution_;
        context_ = other.context_;
    }

    inline ExplicitRungeKutta& operator=(const ExplicitRungeKutta& other)
    {
        *this = ExplicitRungeKutta(other);
        return *this;
    };

    // ExplicitRungeKutta(EquationType eqnSystem, const Dictionary& dict)
    //     : TimeIntegrationFactory::Register<ExplicitRungeKutta>(eqnSystem, dict)
    // {
    //     initSUNERKSolver();
    // }

    static std::string name() { return "explicitRungeKutta"; }

    static std::string doc() { return "Explicit time integration using the Runge-Kutta method."; }

    static std::string schema() { return "none"; }

    void solve(Expression& exp, SolutionFieldType& solutionField, scalar dt) override
    {
        if (data_ == nullptr) initSUNERKSolver();
        data_->system_ =
            std::make_unique<Expression>(exp); // This should be a construction/init thing, but I
                                               //  don't have the equation on construction anymore.

        // Load the current solution for temporal integration
        sunrealtype* solution = N_VGetArrayPointer(solution_);
        auto& field = solutionField.internalField();
        for (size_t i = 0; i < solutionField.size(); ++i)
        {
            solution[i] = field[i];
        }

        data_->fixedStepSize_ = dt;
        auto stepReturn = ARKStepEvolve(
            arkodeMemory_.get(), time_ + data_->fixedStepSize_, solution_, &time_, ARK_ONE_STEP
        );

        // Copy sundials solution back to the solution container.
        for (size_t i = 0; i < solutionField.size(); ++i)
        {
            field[i] = solution[i];
        }

        std::cout << "Step t = " << time_ << "\tcode: " << stepReturn << "\t" << field[0]
                  << std::endl;
    }

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> clone() const
    {
        return std::make_unique<ExplicitRungeKutta>(*this);
    }


private:

    // double timeStepSize_;
    double time_;
    VecType kokkosSolution_;
    VecType kokkosInitialConditions_;
    N_Vector initialConditions_;
    N_Vector solution_;
    SUNContext context_;
    std::unique_ptr<char> arkodeMemory_; // this should be void* but that is not stl compliant we
                                         // store the next best thing.
    std::unique_ptr<NFData> data_;

    void initSUNERKSolver()
    {
        // NOTE CHECK https://sundials.readthedocs.io/en/latest/arkode/Usage/Skeleton.html for order
        // of initialization.
        initNFData();

        // Initialize SUNdials solver;
        initSUNContext();
        initSUNDimension();
        initSUNInitialConditions();
        initSUNCreateERK();
        initSUNTolerances();
    }

    void initNFData()
    {
        data_ = std::make_unique<NFData>();
        data_->realTol_ = this->dict_.template get<scalar>("Relative Tolerance");
        data_->absTol_ = this->dict_.template get<scalar>("Absolute Tolerance");
        data_->fixedStepSize_ =
            this->dict_.template get<scalar>("Fixed Step Size"); // zero for adaptive
        data_->endTime_ = this->dict_.template get<scalar>("End Time");
        data_->nodes = 1;
        data_->maxsteps = 1;
    }

    void initSUNContext()
    {
        int flag = SUNContext_Create(SUN_COMM_NULL, &context_);
        NF_ASSERT(flag == 0, "SUNContext_Create failed");
    }

    void initSUNDimension()
    {
        kokkosSolution_ = VecType(data_->nodes, context_);
        kokkosInitialConditions_ = VecType(data_->nodes, context_);
        solution_ = kokkosSolution_;
        initialConditions_ = kokkosInitialConditions_;
    }

    void initSUNInitialConditions() { N_VConst(1.0, initialConditions_); }

    void initSUNCreateERK()
    {
        arkodeMemory_.reset(reinterpret_cast<char*>(
            ERKStepCreate(explicitSolveWrapperFreeFunction, 0.0, initialConditions_, context_)
        ));
        void* ark = reinterpret_cast<void*>(arkodeMemory_.get());

        // Initialize ERKStep solver
        ERKStepSetUserData(ark, NULL);
        ERKStepSetInitStep(ark, data_->fixedStepSize_);
        ERKStepSetFixedStep(ark, data_->fixedStepSize_);
        ERKStepSetTableNum(ark, ARKODE_FORWARD_EULER_1_1);
        ARKodeSetUserData(ark, data_.get());
    }

    void initSUNTolerances()
    {
        ARKStepSStolerances(arkodeMemory_.get(), data_->realTol_, data_->absTol_);
    }
};

template class ExplicitRungeKutta<finiteVolume::cellCentred::VolumeField<scalar>>;


} // namespace NeoFOAM
