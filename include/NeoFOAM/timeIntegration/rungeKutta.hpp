// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>
#include <memory>

#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/timeIntegration/timeIntegration.hpp"
#include "NeoFOAM/timeIntegration/sundials.hpp"


// Where to put?
template<typename SolutionFieldType>
struct NFData
{
    NFData() = default;
    ~NFData() = default;

    // Need to make move constructors
    NFData(const NFData& other)
    {
        system_ = std::make_unique<NeoFOAM::dsl::Expression>(other.system_->exec()
        ); // system of equations
        solution_ = std::make_unique<SolutionFieldType>(*other.solution_.get());
    }

    std::unique_ptr<NeoFOAM::dsl::Expression> system_ {nullptr}; // system of equations
    std::unique_ptr<SolutionFieldType> solution_ {nullptr};
};

template<typename SolutionFieldType>
int explicitSolveWrapperFreeFunction(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
    // Pointer wrangling
    NFData<SolutionFieldType>* nfData = reinterpret_cast<NFData<SolutionFieldType>*>(user_data);
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
    parallelFor(
        source.exec(),
        {0, source.size()},
        KOKKOS_LAMBDA(const size_t i) { ydotarray[i] = source[i]; }
    );

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
    using VectorType = NeoFOAM::sundials::SKVectorType;
    using SKSizeType = NeoFOAM::sundials::SKSizeType;

public:

    using Base = TimeIntegratorBase<SolutionFieldType>::template Register<
        ExplicitRungeKutta<SolutionFieldType>>;
    using Base::dict_;

    ExplicitRungeKutta() = default;

    ~ExplicitRungeKutta() { SUNContext_Free(&context_); };

    ExplicitRungeKutta(const Dictionary& dict) : Base(dict) {}

    ExplicitRungeKutta(const ExplicitRungeKutta& other)
        : Base(other),
          data_(
              other.data_ ? std::make_unique<NFData<SolutionFieldType>>(*other.data_) : nullptr
          ) // Deep copy of unique_ptr
    {
        solution_ = other.solution_;
        context_ = other.context_;
        time_ = other.time_;
    }

    inline ExplicitRungeKutta& operator=(const ExplicitRungeKutta& other)
    {
        *this = ExplicitRungeKutta(other);
        return *this;
    };

    static std::string name() { return "Runge-Kutta"; }

    static std::string doc() { return "Explicit time integration using the Runge-Kutta method."; }

    static std::string schema() { return "none"; }

    void solve(Expression& exp, SolutionFieldType& solutionField, const scalar dt) override
    {
        if (data_ == nullptr) initSUNERKSolver(exp, solutionField, dt);
        std::cout << "\nHERE!:";
        // Load the current solution for temporal integration
        sunrealtype* solution = N_VGetArrayPointer(solution_);
        auto& field = solutionField.internalField();

        parallelFor(
            field.exec(),
            {0, field.size()},
            KOKKOS_LAMBDA(const size_t i) { solution[i] = field[i]; }
        );

        std::cout << "\nHERE!:";
        void* ark = reinterpret_cast<void*>(arkodeMemory_.get());
        ERKStepSetFixedStep(ark, dt);
        auto stepReturn = ARKStepEvolve(ark, time_ + dt, solution_, &time_, ARK_ONE_STEP);
        std::cout << "\nHERE!:";

        auto fieldData = solutionField.internalField().data();
        parallelFor(
            field.exec(),
            {0, field.size()},
            KOKKOS_LAMBDA(const size_t i) { fieldData[i] = solution[i]; }
        );

        auto f = field.copyToHost();
        std::cout << "Step t = " << time_ << "\tcode: " << stepReturn << "\tfield[0]: " << f[0]
                  << std::endl;
    }

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> clone() const
    {
        return std::make_unique<ExplicitRungeKutta>(*this);
    }


private:

    // double timeStepSize_;
    sunrealtype time_;
    VectorType kokkosSolution_;
    VectorType kokkosInitialConditions_;
    N_Vector initialConditions_;
    N_Vector solution_;
    SUNContext context_;
    std::unique_ptr<char> arkodeMemory_; // this should be void* but that is not stl compliant we
                                         // store the next best thing.
    std::unique_ptr<NFData<SolutionFieldType>> data_;

    void initSUNERKSolver(Expression& exp, SolutionFieldType& solutionField, const scalar dt)
    {
        // NOTE CHECK https://sundials.readthedocs.io/en/latest/arkode/Usage/Skeleton.html for order
        // of initialization.
        initNFData(exp);

        // Initialize SUNdials solver;
        initSUNContext();
        initSUNDimension(solutionField);
        initSUNInitialConditions();
        initSUNCreateERK(dt);
        initSUNTolerances();
    }

    void initNFData(Expression& exp)
    {
        data_ = std::make_unique<NFData<SolutionFieldType>>();
        data_->system_ =
            std::make_unique<Expression>(exp); // This should be a construction/init thing, but I
                                               //  don't have the equation on construction anymore.
    }

    void initSUNContext()
    {
        int flag = SUNContext_Create(SUN_COMM_NULL, &context_);
        NF_ASSERT(flag == 0, "SUNContext_Create failed");
    }

    void initSUNDimension(SolutionFieldType solutionField)
    {
        kokkosSolution_ = VectorType(solutionField.internalField().size(), context_);
        kokkosInitialConditions_ = VectorType(solutionField.internalField().size(), context_);
        solution_ = kokkosSolution_;
        initialConditions_ = kokkosInitialConditions_;
    }

    void initSUNInitialConditions() { N_VConst(1.0, initialConditions_); }

    void initSUNCreateERK(const scalar dt)
    {
        arkodeMemory_.reset(reinterpret_cast<char*>(ERKStepCreate(
            explicitSolveWrapperFreeFunction<SolutionFieldType>, 0.0, initialConditions_, context_
        )));
        void* ark = reinterpret_cast<void*>(arkodeMemory_.get());

        // Initialize ERKStep solver
        ERKStepSetUserData(ark, NULL);
        ERKStepSetInitStep(ark, dt);
        ERKStepSetTableNum(
            ark,
            NeoFOAM::sundials::stringToERKTable(
                this->dict_.template get<std::string>("Runge-Kutta Method")
            )
        );
        ARKodeSetUserData(ark, data_.get());
    }

    void initSUNTolerances()
    {
        ARKStepSStolerances(arkodeMemory_.get(), 1.0, 1.0); // If we want ARK we will revisit.
    }
};

template class ExplicitRungeKutta<finiteVolume::cellCentred::VolumeField<scalar>>;


} // namespace NeoFOAM
