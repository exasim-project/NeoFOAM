// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
// TODO: move to cellCenred dsl?

#pragma once

#include "NeoN/NeoN.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"
#include "NeoFOAM/compatibility/fvSchemes.hpp"

namespace dsl = NeoN::dsl;

namespace NeoFOAM
{

// Non-template helpers for PDE<scalar>::solveImpl. Defined in src/datastructures/pde.cpp so
// the FixedValueConstraints and SetReference SYCL kernels are compiled only once, not in every
// translation unit that instantiates PDE<scalar> (e.g. kOmegaSST.cpp, kEpsilon.cpp, …).
namespace detail
{
using ScalarLinearSystem = NeoN::la::LinearSystem<NeoN::scalar, NeoN::scalar>;
void applyFixedValueConstraints(
    ScalarLinearSystem& ls,
    NeoN::View<const NeoN::scalar> mask,
    NeoN::View<const NeoN::scalar> values,
    NeoN::localIdx nCells
);
void applySetReference(ScalarLinearSystem& ls, NeoN::localIdx refCell, NeoN::scalar refValue);
} // namespace detail

/*@brief extends expression by giving access to assembled matrix
 * @note used in neoIcoFOAM directly instead of dsl::expression
 * TODO: implement flag if matrix is assembled or not -> if not assembled call assemble
 * for dependent operations like discrete momentum fields
 * needs storage for assembled matrix? and whether update is needed like for rAU and HbyA
 */
template<
    typename ValueType,
    typename MatrixValueType = NeoN::scalar,
    typename IndexType = NeoN::localIdx>
class PDE
{
    using VolumeField = NeoN::finiteVolume::cellCentred::VolumeField<ValueType>;
    // Matrix coefficients use MatrixValueType (scalar by default, giving the segregated
    // vector-solve form for Vec3 fields); the rhs/solution use the field's ValueType.
    // TODO: future work selects MatrixValueType == ValueType (the coupled Vec3 matrix)
    // based on the presence of boundary conditions that require the full block-coupled form.
    using LinearSystem = NeoN::la::LinearSystem<MatrixValueType, ValueType>;

public:

    PDE(dsl::Expression<ValueType> expr, VolumeField& psi, RunTime& runTime)
        : psi_(&psi)
        , expr_(expr)
        , runTime_(&runTime)
        , ls_(readOrCreate<LinearSystem>(
              runTime,
              "linearSystem" + psi.name,
              // FIXME find a proper place
              [&psi, &runTime]()
              {
                  if (runTime.fvSolutionDict.subDict("solvers")
                          .subDict(psi.name)
                          .template get<std::string>("assemblyStrategy", "face-based")
                      == "cell-based")
                  {
                      auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
                      return NeoN::la::createEmptyLinearSystem<MatrixValueType, ValueType>(
                          psi.mesh(),
                          cellIterator
                      );
                  }
                  else
                  {
                      return NeoN::la::createEmptyLinearSystem<MatrixValueType, ValueType>(psi.mesh(
                      ));
                  }
              }
          ))
    {
        // TODO run NeoN expr_ = NeoN::dsl::optimize(expr); if optimize is set in fvSolution
        // NOTE OpenFOAM tokenizes a switch like 'optimize true;' as a word, so it is stored
        // as a std::string in the NeoN dictionary; reading it as bool throws bad_any_cast.
        auto optimize = runTime_->fvSolutionDict.subDict("solvers")
                            .subDict(psi_->name)
                            .template get<std::string>("optimize", "false");
        if (optimize == "true" || optimize == "yes" || optimize == "on" || optimize == "1")
        {
            expr_ = NeoN::dsl::optimize(expr_);
        };

        expr_.read(NeoFOAM::expandSchemeDefaults(runTime_->fvSchemesDict, expr_, psi_->name));
    };

    /** @brief Construct from expression only; field and RunTime are injected on first
     *  assemble(psi, rt) or solveWith(...) call via Solver. */
    PDE(dsl::Expression<ValueType> expr)
        : psi_(nullptr)
        , expr_(std::move(expr))
        , runTime_(nullptr)
        , ls_(std::nullopt)
    {}

    PDE(const PDE& expr)
        : psi_(expr.psi_)
        , expr_(expr.expr_)
        , runTime_(expr.runTime_)
        , ls_(expr.ls_) {};

    ~PDE() = default;

    VolumeField& getField() { return *psi_; }

    const VolumeField& getField() const { return *psi_; }

    [[nodiscard]] LinearSystem& linearSystem() { return *ls_; }

    [[nodiscard]] const LinearSystem& linearSystem() const { return *ls_; }

    NeoN::dsl::Expression<ValueType>& expression() { return expr_; }

    const NeoN::Executor& exec() const { return ls_->exec(); }


    NeoN::finiteVolume::cellCentred::DdtScheme ddtScheme() const
    {
        for (const auto& op : expr_.temporalOperators())
        {
            const auto s = op.ddtScheme();
            if (s != NeoN::finiteVolume::cellCentred::DdtScheme::None)
            {
                return s;
            }
        }
        return NeoN::finiteVolume::cellCentred::DdtScheme::None;
    }

    void setReference(NeoN::localIdx pRefCell, NeoN::scalar pRefValue)
    {
        // Accept the call in serial (MPI not initialised — mpiRank is the
        // sentinel -1, which would compare unequal to 0 if cast to size_t)
        // and on rank 0 in parallel. SetReference::operator() further guards
        // the matrix mutation against non-rank-0 in initialised MPI.
        // When runTime_ is null (1-arg constructor path), default to accepting
        // on all ranks — SetReference::operator() guards non-rank-0 internally.
        bool accept = true;
        if (runTime_ != nullptr)
        {
            const auto& mpiEnv = runTime_->mpiEnvironment;
            accept = !mpiEnv.isInitialized() || mpiEnv.rank() == 0;
        }
        if (accept)
        {
            needReference_ = true;
            pRefCell_ = pRefCell;
            pRefValue_ = pRefValue;
        }
    }

    /** @brief When true, selects the <field>Final relaxation factor and solver subdict. */
    void setFinalIter(bool finalIter) { finalIter_ = finalIter; }

    /** @brief Hard-pin a set of cells to prescribed values after assembly (omega wall function). */
    void
    setConstraints(const NeoN::Vector<NeoN::scalar>& mask, const NeoN::Vector<ValueType>& values)
    {
        constraintMask_ = &mask;
        constraintValues_ = &values;
    }

    /** @brief As setConstraints, but the PDE keeps its own copies of the mask/values so the
     *  caller need not keep them alive until solve() — used by the Python one-shot
     *  epsilon-wall-cell pin helper (epsilonWallFunction). */
    void setConstraintsOwned(
        const NeoN::Vector<NeoN::scalar>& mask,
        const NeoN::Vector<ValueType>& values
    )
    {
        ownedConstraintMask_ = mask;
        ownedConstraintValues_ = values;
        constraintMask_ = &ownedConstraintMask_.value();
        constraintValues_ = &ownedConstraintValues_.value();
    }

    /** @brief assemble the linear system owned by the solver based on the current expression */
    LinearSystem& assemble()
    {
        ls_->reset();
        expr_.assemble(runTime_->t, runTime_->dt, *ls_, psi_->mesh());
        return *ls_;
    }

    /** @brief assemble using injected field and runTime (for lazy-init PDE) */
    LinearSystem& assemble(VolumeField& psi, RunTime& rt)
    {
        initIfNeeded(psi, rt);
        ls_->reset();
        expr_.assemble(rt.t, rt.dt, *ls_, psi.mesh());
        return *ls_;
    }

    /** @brief Assemble and relax the owned ls_ without solving.
     *
     * Ensures computeRAUandHByA reads the relaxed diagonal even when the momentum
     * predictor is disabled. With no relaxation factor configured, this is a no-op.
     */
    LinearSystem& assembleAndRelax()
    {
        assemble();
        relaxOwnedLs();
        return *ls_;
    }

    /** @brief assemble the linear system with an additional rhs term
     *
     * the following assembly logic is applied
     * 1. the "owned" linear system gets assembled
     * 2. a copy of the assembled linear system is made and the rhs is assembled
     * 3. the new linear system with rhs is returned
     */
    LinearSystem assemble(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
    {
        auto rhsExpr = dsl::Expression<ValueType>(-1.0 * rhs);
        rhsExpr.read(NeoFOAM::expandSchemeDefaults(runTime_->fvSchemesDict, rhsExpr, psi_->name));
        auto ls = LinearSystem(assemble());
        rhsExpr.assembleExplicitSource(ls, psi_->mesh());

        return ls;
    }

    NeoN::la::SolverStats solve()
    {
        ls_->reset();
        return solveImpl(expr_, *ls_);
    }

    /** @brief solve expression with additional rhs
     *
     * This function will create two versions of the linear system corresponding to the expression
     * 1. the owned linear system without rhs is assembled and stored
     * 2. a temporary linear system with rhs assembled and solved
     */
    NeoN::la::SolverStats solve(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
    {
        // Assemble and relax the owned ls_ so computeRAUandHByA reads the relaxed diagonal.
        // Apply -grad p to ls_->rhs() in place (avoids copying the matrix), snapshot the rhs
        // beforehand and restore it after solve so ls_ retains the H-system for computeRAUandHByA.
        assemble();
        relaxOwnedLs();


        auto rhsExpr = dsl::Expression<ValueType>(-1.0 * rhs);
        rhsExpr.read(NeoFOAM::expandSchemeDefaults(runTime_->fvSchemesDict, rhsExpr, psi_->name));
        auto savedRhs = NeoN::Vector<ValueType>(ls_->rhs());
        rhsExpr.assembleExplicitSource(*ls_, psi_->mesh());

        auto solverDict = runTime_->fvSolutionDict.subDict("solvers");
        const std::string finalKey = psi_->name + "Final";
        auto fvSolution = (finalIter_ && solverDict.isDict(finalKey))
                            ? solverDict.subDict(finalKey)
                            : solverDict.subDict(psi_->name);
        // Drop NeoFOAM-only keys before handing the dict to NeoN/Ginkgo, whose
        // config parser rejects unknown keys (e.g. assemblyStrategy, optimize).
        stripNeoFOAMKeys(fvSolution);
        auto solver = NeoN::la::Solver(psi_->exec(), fvSolution);
        // Do some sanity checks before trying to solve
        // NF_ASSERT(ls.exec() == solution.exec(), "Executors are not the same");
        auto stats = solver.solve(*ls_, psi_->internalVector());

        ls_->rhs() = savedRhs;

        reportSolverStats(stats, fvSolution);
        return stats;
    }

    /** @brief solve using a pre-created, cached NeoN::la::Solver */
    NeoN::la::SolverStats solveWith(NeoN::la::Solver& solver, VolumeField& psi, RunTime& rt)
    {
        initIfNeeded(psi, rt);

        expr_.read(NeoFOAM::expandSchemeDefaults(rt.fvSchemesDict, expr_, psi.name));
        ls_->reset();
        expr_.assemble(rt.t, rt.dt, *ls_, psi.mesh());

        const auto alpha =
            lookupEqnRelaxation(rt.fvSolutionDict, psi.name, finalIter_).value_or(1.0);
        NeoN::dsl::applyMatrixRelaxation(*ls_, psi, alpha);

        if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
        {
            if (needReference_)
            {
                NeoN::dsl::SetReference<ValueType> refFunct(pRefCell_, pRefValue_);
                refFunct(*ls_);
            }
        }

        auto solverDict = rt.fvSolutionDict.subDict("solvers");
        const std::string finalKey = psi.name + "Final";
        const bool useFinal = finalIter_ && solverDict.isDict(finalKey);
        auto fieldSolverDict =
            useFinal ? solverDict.subDict(finalKey) : solverDict.subDict(psi.name);
        stripNeoFOAMKeys(fieldSolverDict);
        NeoN::fence(psi.exec());
        NF_ASSERT(ls_->exec() == psi.exec(), "Executors are not the same");
        NeoN::la::SolverStats stats;
        if (useFinal)
        {
            NeoN::la::Solver finalSolver(psi.exec(), fieldSolverDict);
            stats = finalSolver.solve(*ls_, psi.internalVector());
        }
        else
        {
            stats = solver.solve(*ls_, psi.internalVector());
        }

        reportSolverStats(stats, fieldSolverDict);
        return stats;
    }

    /** @brief solve with additional rhs using a pre-created, cached NeoN::la::Solver */
    NeoN::la::SolverStats solveWith(
        NeoN::la::Solver& solver,
        VolumeField& psi,
        RunTime& rt,
        dsl::SpatialOperator<ValueType>&& rhs
    )
    {
        initIfNeeded(psi, rt);

        expr_.read(NeoFOAM::expandSchemeDefaults(rt.fvSchemesDict, expr_, psi.name));
        ls_->reset();
        expr_.assemble(rt.t, rt.dt, *ls_, psi.mesh());

        const auto alpha =
            lookupEqnRelaxation(rt.fvSolutionDict, psi.name, finalIter_).value_or(1.0);
        NeoN::dsl::applyMatrixRelaxation(*ls_, psi, alpha);

        // add rhs in place; save and restore so ls_ retains the H-system
        auto rhsExpr = dsl::Expression<ValueType>(-1.0 * rhs);
        rhsExpr.read(NeoFOAM::expandSchemeDefaults(rt.fvSchemesDict, rhsExpr, psi.name));
        auto savedRhs = NeoN::Vector<ValueType>(ls_->rhs());
        rhsExpr.assembleExplicitSource(*ls_, psi.mesh());

        auto solverDict = rt.fvSolutionDict.subDict("solvers");
        const std::string finalKey = psi.name + "Final";
        const bool useFinal = finalIter_ && solverDict.isDict(finalKey);
        auto fieldSolverDict =
            useFinal ? solverDict.subDict(finalKey) : solverDict.subDict(psi.name);
        stripNeoFOAMKeys(fieldSolverDict);
        NeoN::fence(psi.exec());
        NF_ASSERT(ls_->exec() == psi.exec(), "Executors are not the same");
        NeoN::la::SolverStats stats;
        if (useFinal)
        {
            NeoN::la::Solver finalSolver(psi.exec(), fieldSolverDict);
            stats = finalSolver.solve(*ls_, psi.internalVector());
        }
        else
        {
            stats = solver.solve(*ls_, psi.internalVector());
        }

        ls_->rhs() = savedRhs;

        reportSolverStats(stats, fieldSolverDict);
        return stats;
    }

    // Public because NVCC forbids extended __host__ __device__ lambdas
    // (NEON_LAMBDA) inside private or protected member functions.
    NeoN::la::SolverStats solveImpl(dsl::Expression<ValueType>& expr, LinearSystem& ls)
    {
        // Re-read schemes (idempotent with the constructor read)
        expr.read(NeoFOAM::expandSchemeDefaults(runTime_->fvSchemesDict, expr, psi_->name));

        // Assemble without post-assembly functors; we apply SetReference separately below
        // to ensure correct polymorphic dispatch — storing PostAssemblyBase by value causes
        // object slicing that silently disables virtual overrides.
        expr.assemble(runTime_->t, runTime_->dt, ls, psi_->mesh());

        // Relaxation MUST precede SetReference: SetReference doubles the ref-cell diagonal,
        // and relaxing afterwards would corrupt the pin.
        const auto alpha =
            lookupEqnRelaxation(runTime_->fvSolutionDict, psi_->name, finalIter_).value_or(1.0);
        NeoN::dsl::applyMatrixRelaxation(ls, *psi_, alpha);

        // Apply reference-cell pinning (kernel compiled only in pde.cpp via detail helper)
        if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
        {
            if (needReference_)
            {
                detail::applySetReference(ls, pRefCell_, pRefValue_);
            }
        }

        if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
        {
            if (constraintMask_ != nullptr)
            {
                const NeoN::localIdx nCells = psi_->mesh().nCells();
                detail::applyFixedValueConstraints(
                    ls,
                    constraintMask_->view(),
                    constraintValues_->view(),
                    nCells
                );
            }
        }

        auto solverDict = runTime_->fvSolutionDict.subDict("solvers");
        const std::string finalKey = psi_->name + "Final";
        auto fieldSolverDict = (finalIter_ && solverDict.isDict(finalKey))
                                 ? solverDict.subDict(finalKey)
                                 : solverDict.subDict(psi_->name);
        // Drop NeoFOAM-only keys before handing the dict to NeoN/Ginkgo, whose
        // config parser rejects unknown keys (e.g. assemblyStrategy, optimize).
        stripNeoFOAMKeys(fieldSolverDict);
        NeoN::fence(psi_->exec());
        NF_ASSERT(ls.exec() == psi_->exec(), "Executors are not the same");

        auto solver = NeoN::la::Solver(psi_->exec(), fieldSolverDict);
        auto stats = solver.solve(ls, psi_->internalVector());

        reportSolverStats(stats, fieldSolverDict);

        return stats;
    }

    // Relax the owned ls_ in place. Shared by solve(rhs) and assembleAndRelax() so both
    // paths use the same alpha lookup. Must run before SetReference (see solveImpl).
    // Only calls the free function applyMatrixRelaxation — no NEON_LAMBDA in this body.
    void relaxOwnedLs()
    {
        const auto alpha =
            lookupEqnRelaxation(runTime_->fvSolutionDict, psi_->name, finalIter_).value_or(1.0);
        NeoN::dsl::applyMatrixRelaxation(*ls_, *psi_, alpha);
    }

    // Remove NeoFOAM-specific control keys from a per-field solver dict before it
    // is passed to NeoN/Ginkgo. Ginkgo's config parser is strict and aborts on any
    // unrecognised key, so these must be popped here rather than left in place.
    static void stripNeoFOAMKeys(NeoN::Dictionary& dict)
    {
        for (const auto& key : {"assemblyStrategy", "optimize"})
        {
            if (dict.contains(key))
            {
                dict.remove(key);
            }
        }
    }

private:

    /** @brief initialize field and runTime on first use (1-arg constructor path) */
    void initIfNeeded(VolumeField& psi, RunTime& rt)
    {
        if (psi_ != nullptr) return;

        psi_ = &psi;
        runTime_ = &rt;

        auto optimize =
            rt.fvSolutionDict.subDict("solvers").subDict(psi.name).template get<std::string>(
                "optimize",
                "false"
            );
        if (optimize == "true" || optimize == "yes" || optimize == "on" || optimize == "1")
        {
            expr_ = NeoN::dsl::optimize(expr_);
        }
        expr_.read(NeoFOAM::expandSchemeDefaults(rt.fvSchemesDict, expr_, psi.name));

        ls_.emplace(readOrCreate<LinearSystem>(
            rt,
            "linearSystem" + psi.name,
            [&psi, &rt]()
            {
                if (rt.fvSolutionDict.subDict("solvers")
                        .subDict(psi.name)
                        .template get<std::string>("assemblyStrategy", "face-based")
                    == "cell-based")
                {
                    auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
                    return NeoN::la::createEmptyLinearSystem<MatrixValueType, ValueType>(
                        psi.mesh(),
                        cellIterator
                    );
                }
                else
                {
                    return NeoN::la::createEmptyLinearSystem<MatrixValueType, ValueType>(psi.mesh()
                    );
                }
            }
        ));
    }

    // Per-component name (Ux/Uy/Uz) when a vector field is solved as separate
    // component systems; the plain field name otherwise (scalar or coupled solve).
    static std::string componentName(const std::string& base, std::size_t i, std::size_t n)
    {
        if (n <= 1) return base;
        constexpr const char* suffix[3] = {"x", "y", "z"};
        return i < 3 ? base + suffix[i] : base + "[" + std::to_string(i) + "]";
    }

    // OpenFOAM-style per-solve residual report: "<precond><solver>:  Solving for
    // <field>, Initial residual = ..., Final residual = ..., No Iterations N".
    // The "<precond><solver>" label (e.g. DICPCG) is read from the solver dict's
    // reportName meta key stashed by mapFvSolution -> no recomputation per solve.
    // Logging is rank-aware and skips formatting on muted ranks (NeoN shouldLog).
    void reportSolverStats(
        const NeoN::la::SolverStats& stats,
        const NeoN::Dictionary& fieldSolverDict
    ) const
    {
        const std::string label = fieldSolverDict.contains("reportName")
                                    ? fieldSolverDict.get<std::string>("reportName")
                                    : std::string("Ginkgo");
        const std::size_t n = stats.entries.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto& stat = stats.entries[i];
            NeoN::Logging::info(
                "{}:  Solving for {}, Initial residual = {}, Final residual = {}, No Iterations {}",
                label,
                componentName(psi_->name, i, n),
                stat.initResNorm,
                stat.finalResNorm,
                stat.numIter
            );
        }
    }

    VolumeField* psi_;
    dsl::Expression<ValueType> expr_;
    RunTime* runTime_;
    std::optional<LinearSystem> ls_;
    bool needReference_ = false;
    NeoN::localIdx pRefCell_ = 0;
    NeoN::scalar pRefValue_ = 0.0;
    bool finalIter_ = false;
    const NeoN::Vector<NeoN::scalar>* constraintMask_ = nullptr;
    const NeoN::Vector<ValueType>* constraintValues_ = nullptr;
    // Optional PDE-owned storage backing constraintMask_/constraintValues_ (setConstraintsOwned).
    std::optional<NeoN::Vector<NeoN::scalar>> ownedConstraintMask_;
    std::optional<NeoN::Vector<ValueType>> ownedConstraintValues_;
};


template<typename ValueType>
using PDESolver = PDE<ValueType>;

template<typename ValueType, typename IndexType = NeoN::localIdx>
NeoN::finiteVolume::cellCentred::VolumeField<ValueType> applyOperator(
    const la::LinearSystem<ValueType>& ls,
    const NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& psi
)
{
    NeoN::finiteVolume::cellCentred::VolumeField<ValueType> res(
        psi.exec(),
        "ls_" + psi.name,
        psi.mesh(),
        psi.internalVector(),
        psi.boundaryData(),
        psi.boundaryConditions()
    );
    NeoN::la::computeResidual(ls.matrix(), ls.rhs(), psi.internalVector(), res.internalVector());
    return res;
}


template<typename ValueType>
NeoN::finiteVolume::cellCentred::VolumeField<ValueType> operator&(
    const PDE<ValueType>& expr,
    const NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& psi
)
{
    return applyOperator(expr.linearSystem(), psi);
}

}
