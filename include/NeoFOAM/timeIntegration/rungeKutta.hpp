// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>
#include <memory>

#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/timeIntegration/timeIntegration.hpp"
#include "NeoFOAM/timeIntegration/sundials.hpp"

namespace NeoFOAM::dsl
{

// as per sundails manual, once we create kokkos vectors and convert to N_Vectors we only
// interact with the N_vectors
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
          PDEExpr_(
              other.PDEExpr_ ? std::make_unique<NeoFOAM::dsl::Expression>(other.PDEExpr_->exec())
                             : nullptr
          )
    {
        solution_ = other.solution_;
        context_ = other.context_;
    }

    inline ExplicitRungeKutta& operator=(const ExplicitRungeKutta& other)
    {
        *this = ExplicitRungeKutta(other);
        return *this;
    };

    static std::string name() { return "Runge-Kutta"; }

    static std::string doc() { return "Explicit time integration using the Runge-Kutta method."; }

    static std::string schema() { return "none"; }

    void
    solve(Expression& exp, SolutionFieldType& solutionField, scalar t, const scalar dt) override
    {
        // Setup sundials if required, load the current solution for temporal integration
        if (PDEExpr_ == nullptr) initSUNERKSolver(exp, solutionField, t, dt);
        NeoFOAM::sundials::fieldToNVector(solutionField.internalField(), solution_);
        void* ark = reinterpret_cast<void*>(arkodeMemory_.get());

        // Perform time integration
        ERKStepSetFixedStep(ark, dt);
        NeoFOAM::scalar timeOut;
        auto stepReturn = ARKStepEvolve(ark, t + dt, solution_, &timeOut, ARK_ONE_STEP);

        // Post step checks
        NF_ASSERT_EQUAL(stepReturn, 0);
        NF_ASSERT_EQUAL(t + dt, timeOut);

        // Copy solution out. (Fence is in sundails free)
        NeoFOAM::sundials::NVectorToField(solution_, solutionField.internalField());
    }

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> clone() const
    {
        return std::make_unique<ExplicitRungeKutta>(*this);
    }


private:

    VectorType kokkosSolution_;
    VectorType kokkosInitialConditions_;
    N_Vector initialConditions_;
    N_Vector solution_;
    SUNContext context_;
    std::unique_ptr<char> arkodeMemory_; // this should be void* but that is not stl compliant we
                                         // store the next best thing.
    std::unique_ptr<NeoFOAM::dsl::Expression> PDEExpr_;

    void initSUNERKSolver(
        Expression& exp, SolutionFieldType& solutionField, const scalar t, const scalar dt
    )
    {
        // NOTE CHECK https://sundials.readthedocs.io/en/latest/arkode/Usage/Skeleton.html for order
        // of initialization.
        initExpression(exp);

        // Initialize SUNdials solver;
        initSUNContext();
        initSUNDimension(solutionField.internalField().size());
        initSUNInitialConditions(solutionField);
        initSUNCreateERK(t, dt);
        initSUNTolerances();
    }

    void initExpression(Expression& exp)
    {
        PDEExpr_ =
            std::make_unique<Expression>(exp); // This should be a construction/init thing, but I
                                               //  don't have the equation on construction anymore.
    }

    void initSUNContext()
    {
        int flag = SUNContext_Create(SUN_COMM_NULL, &context_);
        NF_ASSERT(flag == 0, "SUNContext_Create failed");
    }

    void initSUNDimension(size_t size)
    {
        // see
        // https://sundials.readthedocs.io/en/latest/nvectors/NVector_links.html#the-nvector-kokkos-module
        kokkosSolution_ = VectorType(size, context_);
        kokkosInitialConditions_ = VectorType(size, context_);
        solution_ = kokkosSolution_;
        initialConditions_ = kokkosInitialConditions_;
    }

    void initSUNInitialConditions(SolutionFieldType solutionField)
    {
        NeoFOAM::sundials::fieldToNVector(solutionField.internalField(), initialConditions_);
    }

    void initSUNCreateERK(const scalar t, const scalar dt)
    {
        arkodeMemory_.reset(reinterpret_cast<char*>(ERKStepCreate(
            NeoFOAM::sundials::explicitRKSolve<SolutionFieldType>, t, initialConditions_, context_
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
        ARKodeSetUserData(ark, PDEExpr_.get());
    }

    void initSUNTolerances()
    {
        ARKStepSStolerances(arkodeMemory_.get(), 1.0, 1.0); // If we want ARK we will revisit.
    }
};

template class ExplicitRungeKutta<finiteVolume::cellCentred::VolumeField<scalar>>;


} // namespace NeoFOAM
