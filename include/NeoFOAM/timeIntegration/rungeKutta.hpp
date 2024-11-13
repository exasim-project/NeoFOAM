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
class RungeKutta :
    public TimeIntegratorBase<SolutionFieldType>::template Register<RungeKutta<SolutionFieldType>>
{
    using VectorType = NeoFOAM::sundials::SKVectorType;
    using SKSizeType = NeoFOAM::sundials::SKSizeType;

public:

    using Base =
        TimeIntegratorBase<SolutionFieldType>::template Register<RungeKutta<SolutionFieldType>>;
    using Base::dict_;

    RungeKutta() = default;

    ~RungeKutta() { SUNContext_Free(&context_); };

    RungeKutta(const Dictionary& dict) : Base(dict) {}

    RungeKutta(const RungeKutta& other)
        : Base(other),
          PDEExpr_(
              other.PDEExpr_ ? std::make_unique<NeoFOAM::dsl::Expression>(other.PDEExpr_->exec())
                             : nullptr
          )
    {
        solution_ = other.solution_;
        context_ = other.context_;
    }

    /**
     * @brief Move Constructor
     * @brief other The
     */
    RungeKutta(RungeKutta&& other)
        : Base(std::move(other)), kokkosSolution_(std::move(other.kokkosSolution_)),
          kokkosInitialConditions_(std::move(other.kokkosInitialConditions_)),
          initialConditions_(std::move(other.initialConditions_)),
          solution_(std::move(other.solution_)), context_(std::move(other.context_)),
          arkodeMemory_(std::move(other.arkodeMemory_)), PDEExpr_(std::move(other.PDEExpr_))
    {}


    inline RungeKutta& operator=(const RungeKutta& other)
    {
        if (this == &other) return *this;
        *this = RungeKutta(other);
        return *this;
    };

    RungeKutta& operator=(RungeKutta&& other)
    {
        if (this == &other) return *this;
    }

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
        ARKodeSetFixedStep(ark, dt);
        NeoFOAM::scalar timeOut;
        auto stepReturn = ARKodeEvolve(ark, t + dt, solution_, &timeOut, ARK_ONE_STEP);

        // Post step checks
        NF_ASSERT_EQUAL(stepReturn, 0);
        NF_ASSERT_EQUAL(t + dt, timeOut);

        // Copy solution out. (Fence is in sundails free)
        NeoFOAM::sundials::NVectorToField(solution_, solutionField.internalField());
    }

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> clone() const
    {
        return std::make_unique<RungeKutta>(*this);
    }


private:

    VectorType kokkosSolution_ {};
    VectorType kokkosInitialConditions_ {};
    N_Vector initialConditions_ {nullptr};
    N_Vector solution_ {nullptr};
    SUNContext context_ {};
    std::unique_ptr<char> arkodeMemory_ {nullptr}; // this should be void* but that is not stl
                                                   // compliant, we store the next best thing.
    std::unique_ptr<NeoFOAM::dsl::Expression> PDEExpr_ {nullptr};

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
        ARKodeSetUserData(ark, NULL);
        ARKodeSetInitStep(ark, dt);
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
        ARKodeSStolerances(arkodeMemory_.get(), 1.0, 1.0); // If we want ARK we will revisit.
    }
};

template class RungeKutta<finiteVolume::cellCentred::VolumeField<scalar>>;


} // namespace NeoFOAM
