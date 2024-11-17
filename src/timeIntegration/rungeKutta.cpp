// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/timeIntegration/rungeKutta.hpp"

namespace NeoFOAM::dsl
{

template<typename SolutionFieldType>
RungeKutta<SolutionFieldType>::RungeKutta(const RungeKutta<SolutionFieldType>& other)
    : Base(other), kokkosSolution_(other.kokkosSolution_),
      kokkosInitialConditions_(other.kokkosInitialConditions_), solution_(kokkosSolution_),
      initialConditions_(kokkosInitialConditions_),
      pdeExpr_(
          other.pdeExpr_ ? std::make_unique<NeoFOAM::dsl::Expression>(other.pdeExpr_->exec())
                         : nullptr
      )
{
    sunrealtype timeCurrent;
    void* ark = reinterpret_cast<void*>(other.ODEMemory_.get());
    ARKodeGetCurrentTime(ark, &timeCurrent);
    initODEMemory(timeCurrent); // will finalise construction of the ode memory.
}

template<typename SolutionFieldType>
RungeKutta<SolutionFieldType>::RungeKutta(RungeKutta<SolutionFieldType>&& other)
    : Base(std::move(other)), kokkosSolution_(std::move(other.kokkosSolution_)),
      kokkosInitialConditions_(std::move(other.kokkosInitialConditions_)),
      solution_(std::move(other.solution_)), context_(std::move(other.context_)),
      initialConditions_(std::move(other.initialConditions_)),
      ODEMemory_(std::move(other.ODEMemory_)), pdeExpr_(std::move(other.pdeExpr_))
{}


template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::solve(
    Expression& exp, SolutionFieldType& solutionField, scalar t, const scalar dt
)
{
    // Setup sundials if required, load the current solution for temporal integration
    if (pdeExpr_ == nullptr) initSUNERKSolver(exp, solutionField, t);
    NeoFOAM::sundials::fieldToNVector(solutionField.internalField(), solution_);
    void* ark = reinterpret_cast<void*>(ODEMemory_.get());

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

template<typename SolutionFieldType>
std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> RungeKutta<SolutionFieldType>::clone() const
{
    return std::make_unique<RungeKutta>(*this);
}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initSUNERKSolver(
    Expression& exp, SolutionFieldType& field, const scalar t
)
{
    initExpression(exp);
    initSUNContext();
    initSUNVector(field.internalField().size());
    initSUNInitialConditions(field);
    initODEMemory(t);
}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initExpression(const Expression& exp)
{
    pdeExpr_ =
        std::make_unique<Expression>(exp); // This should be a construction/init thing, but I
                                           //  don't have the equation on construction anymore.
}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initSUNContext()
{
    if (!context_)
    {
        SUNContext rawContext;
        int flag = SUNContext_Create(SUN_COMM_NULL, &rawContext);
        NF_ASSERT(flag == 0, "SUNContext_Create failed");
        context_.reset(rawContext, sundials::SUN_CONTEXT_DELETER);
    }
}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initSUNVector(size_t size)
{
    NF_DEBUG_ASSERT(context_, "SUNContext is a nullptr.");
    kokkosSolution_ = VectorType(size, context_.get());
    kokkosInitialConditions_ = VectorType(size, context_.get());
    solution_ = kokkosSolution_;
    initialConditions_ = kokkosInitialConditions_;
}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initSUNInitialConditions(SolutionFieldType solutionField)
{
    NeoFOAM::sundials::fieldToNVector(solutionField.internalField(), initialConditions_);
}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initODEMemory(const scalar t)
{
    NF_DEBUG_ASSERT(context_, "SUNContext is a nullptr.");
    NF_DEBUG_ASSERT(pdeExpr_, "PDE expression is a nullptr.");

    ODEMemory_.reset(reinterpret_cast<char*>(ERKStepCreate(
        NeoFOAM::sundials::explicitRKSolve<SolutionFieldType>, t, initialConditions_, context_.get()
    )));
    void* ark = reinterpret_cast<void*>(ODEMemory_.get());

    // Initialize ERKStep solver
    ERKStepSetTableNum(
        ark,
        NeoFOAM::sundials::stringToERKTable(
            this->dict_.template get<std::string>("Runge-Kutta Method")
        )
    );
    ARKodeSetUserData(ark, pdeExpr_.get());
    ARKodeSStolerances(ODEMemory_.get(), 1.0, 1.0); // If we want ARK we will revisit.
}

template class RungeKutta<finiteVolume::cellCentred::VolumeField<scalar>>;

}
