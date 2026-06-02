// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
// TODO: move to cellCenred dsl?

#pragma once

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"

namespace dsl = NeoN::dsl;

namespace NeoFOAM
{

/*@brief extends expression by giving access to assembled matrix
 * @note used in neoIcoFOAM directly instead of dsl::expression
 * TODO: implement flag if matrix is assembled or not -> if not assembled call assemble
 * for dependent operations like discrete momentum fields
 * needs storage for assembled matrix? and whether update is needed like for rAU and HbyA
 */
template<typename ValueType, typename IndexType = NeoN::localIdx>
class PDESolver
{
    using VolumeField = NeoN::finiteVolume::cellCentred::VolumeField<ValueType>;
    using LinearSystem = NeoN::la::LinearSystem<ValueType>;

public:

    PDESolver(dsl::Expression<ValueType> expr, VolumeField& psi, RunTime& runTime)
        : psi_(psi)
        , expr_(expr)
        , runTime_(runTime)
        , ls_(readOrCreate<LinearSystem>(
              runTime,
              "linearSystem" + psi.name,
              // FIXME find a proper place
              [&psi, &runTime]()
              { return NeoN::la::createEmptyLinearSystem<ValueType>(psi.mesh()); }
          ))
    {
        expr_.read(runTime_.fvSchemesDict);
    };

    PDESolver(const PDESolver& expr)
        : psi_(expr.psi_)
        , expr_(expr.expr_)
        , runTime_(expr.runTime_)
        , ls_(expr.ls_) {};

    ~PDESolver() = default;

    VolumeField& getField() { return this->psi_; }

    const VolumeField& getField() const { return this->psi_; }

    [[nodiscard]] LinearSystem& linearSystem() { return ls_; }

    [[nodiscard]] const LinearSystem& linearSystem() const { return ls_; }

    NeoN::dsl::Expression<ValueType>& expression() { return expr_; }

    const NeoN::Executor& exec() const { return ls_.exec(); }


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
        const auto& mpiEnv = runTime_.mpiEnvironment;
        if (!mpiEnv.isInitialized() || mpiEnv.rank() == 0)
        {
            needReference_ = true;
            pRefCell_ = pRefCell;
            pRefValue_ = pRefValue;
        }
    }

    /** @brief assemble the linear system owned by the solver based on the current expression */
    LinearSystem& assemble()
    {
        ls_.reset();
        expr_.assemble(runTime_.t, runTime_.dt, ls_);
        return ls_;
    }

    /** @brief assemble the linear system with an additional rhs term
     *
     * the following assembly logic is applied
     * 1. the "owned" linear system gets assembled
     * 2. a copy of the assembled linear system is made and the rhs is assembled
     * 3. the new linear system with rhs is returned
     */
    NeoN::la::LinearSystem<ValueType> assemble(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
    {
        auto rhsExpr = dsl::Expression<ValueType>(-1.0 * rhs);
        rhsExpr.read(runTime_.fvSchemesDict);
        auto ls = NeoN::la::LinearSystem<ValueType>(assemble());

        // Collect explicit contributions from the main expression first, then from the extra rhs.
        // The main expression's explicit ops are subtracted from rhs (DSL convention), so they
        // must be included here just as iterativeSolveImpl does for the no-rhs overload.
        auto expTmp = expr_.explicitOperation(psi_.mesh().nCells());
        rhsExpr.explicitOperation(expTmp);

        auto [vol, expSource, rhsV] = NeoN::views(psi_.mesh().cellVolumes(), expTmp, ls.rhs());
        NeoN::parallelFor(
            psi_.exec(),
            {0, rhsV.size()},
            NEON_LAMBDA(const NeoN::localIdx i) { rhsV[i] -= expSource[i] * vol[i]; }
        );

        return ls;
    }

    NeoN::la::SolverStats solve()
    {
        ls_.reset();
        return solveImpl(expr_, ls_);
    }

    /** @brief solve expression with additional rhs
     *
     * This function will create two versions of the linear system corresponding to the expression
     * 1. the owned linear system without rhs is assembled and stored
     * 2. a temporary linear system with rhs assembled and solved
     */
    NeoN::la::SolverStats solve(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
    {
        // assemble wo rhs first
        auto ls = assemble(std::move(rhs));

        auto solverDict = runTime_.fvSolutionDict.subDict("solvers");
        auto fvSolution = solverDict.subDict(psi_.name);
        auto solver = NeoN::la::Solver(psi_.exec(), fvSolution);
        // Do some sanity checks before trying to solve
        // NF_ASSERT(ls.exec() == solution.exec(), "Executors are not the same");
        auto stats = solver.solve(ls, psi_.internalVector());

        reportSolverStats(stats, fvSolution);
        return stats;
    }

    // Public because NVCC forbids extended __host__ __device__ lambdas
    // (NEON_LAMBDA) inside private or protected member functions.
    NeoN::la::SolverStats solveImpl(dsl::Expression<ValueType>& expr, LinearSystem& ls)
    {
        // Re-read schemes (idempotent with the constructor read)
        expr.read(runTime_.fvSchemesDict);

        // Assemble without post-assembly functors; we apply SetReference separately below
        // to ensure correct polymorphic dispatch — storing PostAssemblyBase by value causes
        // object slicing that silently disables virtual overrides.
        expr.assemble(runTime_.t, runTime_.dt, ls);

        // Subtract the explicit source term from the rhs (mirrors iterativeSolveImpl)
        auto expTmp = expr.explicitOperation(psi_.mesh().nCells());
        auto [vol, expSource, rhs] = NeoN::views(psi_.mesh().cellVolumes(), expTmp, ls.rhs());
        NeoN::parallelFor(
            psi_.exec(),
            {0, static_cast<NeoN::localIdx>(rhs.size())},
            NEON_LAMBDA(const NeoN::localIdx i) { rhs[i] -= expSource[i] * vol[i]; }
        );

        // Apply reference-cell pinning directly (avoids object-slicing issue)
        if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
        {
            if (needReference_)
            {
                NeoN::dsl::SetReference<ValueType> refFunct(pRefCell_, pRefValue_);
                refFunct(ls);
            }
        }

        auto solverDict = runTime_.fvSolutionDict.subDict("solvers");
        auto fieldSolverDict = solverDict.subDict(psi_.name);
        NeoN::fence(psi_.exec());
        NF_ASSERT(ls.exec() == psi_.exec(), "Executors are not the same");

        auto solver = NeoN::la::Solver(psi_.exec(), fieldSolverDict);
        auto stats = solver.solve(ls, psi_.internalVector());

        reportSolverStats(stats, fieldSolverDict);

        return stats;
    }

private:

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
    void reportSolverStats(const NeoN::la::SolverStats& stats, const NeoN::Dictionary& fieldSolverDict)
        const
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
                componentName(psi_.name, i, n),
                stat.initResNorm,
                stat.finalResNorm,
                stat.numIter
            );
        }
    }

    VolumeField& psi_;
    dsl::Expression<ValueType> expr_;
    const RunTime& runTime_;
    LinearSystem ls_;
    bool needReference_ = false;
    NeoN::localIdx pRefCell_ = 0;
    NeoN::scalar pRefValue_ = 0.0;
};


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


template<typename ValueType, typename IndexType = NeoN::localIdx>
NeoN::finiteVolume::cellCentred::VolumeField<ValueType> operator&(
    const PDESolver<ValueType, IndexType> expr,
    const NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& psi
)
{
    return applyOperator(expr.linearSystem(), psi);
}

}
