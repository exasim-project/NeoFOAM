// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
// TODO: move to cellCenred dsl?

#pragma once

#include "NeoN/NeoN.hpp"

#include <algorithm>
#include <cmath>
#include <string>

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
template<
    typename ValueType,
    typename MatrixValueType = NeoN::scalar,
    typename IndexType = NeoN::localIdx>
class PDESolver
{
    using VolumeField = NeoN::finiteVolume::cellCentred::VolumeField<ValueType>;
    // Matrix coefficients use MatrixValueType (scalar by default, giving the segregated
    // vector-solve form for Vec3 fields); the rhs/solution use the field's ValueType.
    // TODO: future work selects MatrixValueType == ValueType (the coupled Vec3 matrix)
    // based on the presence of boundary conditions that require the full block-coupled form.
    using LinearSystem = NeoN::la::LinearSystem<MatrixValueType, ValueType>;

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
        auto optimize =
            runTime_.fvSolutionDict.subDict("solvers").subDict(psi_.name).template get<std::string>(
                "optimize",
                "false"
            );
        if (optimize == "true" || optimize == "yes" || optimize == "on" || optimize == "1")
        {
            expr_ = NeoN::dsl::optimize(expr_);
        };

        expr_.read(runTime_.fvSchemesDict);
    };

    PDESolver(const PDESolver& expr)
        : psi_(expr.psi_)
        , expr_(expr.expr_)
        , runTime_(expr.runTime_)
        , ls_(expr.ls_)
        , matrixReuse_(expr.matrixReuse_)
        , matrixAssembled_(false) {};

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
        expr_.assemble(runTime_.t, runTime_.dt, ls_, psi_.mesh());
        matrixAssembled_ = true;
        return ls_;
    }

    /** @brief enable reusing the assembled matrix across solves of this PDESolver, refreshing only
     * the rhs each solve instead of re-assembling the matrix.
     *
     * Valid when the implicit matrix coefficients are unchanged between solves — e.g. the pressure
     * Poisson matrix laplacian(rAU, p) across non-orthogonal / PISO correctors on a static mesh,
     * where only the deferred non-orthogonal correction and the explicit div(phiHbyA) rhs change.
     * Call markMatrixDirty() whenever the matrix must be rebuilt (moving mesh, changed rAU /
     * diffusivity, changed implicit BC coefficients). Only effective for the non-segregated
     * (scalar-matrix == field-type) form; the segregated vector solve always re-assembles, and a
     * setReference pin forces a full assemble as well.
     */
    void enableMatrixReuse(bool on = true)
    {
        matrixReuse_ = on;
        matrixAssembled_ = false; // force a full assemble on the next solve
    }

    /** @brief invalidate the cached matrix so the next solve re-assembles it in full. Hook for
     * moving mesh / changed coefficients while matrix reuse is enabled. */
    void markMatrixDirty() { matrixAssembled_ = false; }

    NeoN::la::SolverStats solve() { return solveImpl(expr_, ls_); }

    /** @brief solve the expression augmented with an additional explicit rhs term
     * (e.g. the momentum predictor's -grad(p)).
     *
     * The owned system ls_ is assembled once and solved in place: the extra rhs term is applied to
     * ls_.rhs() only for the duration of the solve and then added back, so ls_ is left holding the
     * rhs-free ("H") system the subsequent rAU/HbyA step reads. With the momentum predictor enabled
     * computeRAUandHByA consumes ls_.rhs() directly (the apps do not re-assemble in that branch),
     * so the restore is required for a correct H. This avoids deep-copying the whole momentum
     * LinearSystem on every solve; the solver takes the system rhs as const, so re-adding the
     * source reproduces the previous copy-based behaviour exactly.
     */
    NeoN::la::SolverStats solve(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
    {
        // 1. assemble the owned system once (implicit + main explicit sources via NeoN assemble).
        assemble();

        // 2. build the extra rhs term (e.g. -grad(p)). The main expression's explicit sources are
        //    already folded into ls_ by assemble(), so only the extra term is collected here.
        auto rhsExpr = dsl::Expression<ValueType>(-1.0 * rhs);
        rhsExpr.read(runTime_.fvSchemesDict);
        auto expSource =
            rhsExpr.explicitOperation(static_cast<NeoN::localIdx>(psi_.mesh().nCells()));

        // 3. apply the explicit source to ls_.rhs() in place for the solve (rhs -= source * vol).
        {
            auto [vol, src, rhsV] = NeoN::views(psi_.mesh().cellVolumes(), expSource, ls_.rhs());
            NeoN::parallelFor(
                psi_.exec(),
                {0, rhsV.size()},
                NEON_LAMBDA(const NeoN::localIdx i) { rhsV[i] -= src[i] * vol[i]; }
            );
        }

        auto solverDict = runTime_.fvSolutionDict.subDict("solvers");
        auto fvSolution = solverDict.subDict(psi_.name);
        stripNeoFOAMKeys(fvSolution);
        auto solver = NeoN::la::Solver(psi_.exec(), fvSolution);
        auto stats = solver.solve(ls_, psi_.internalVector());
        reportSolverStats(stats, fvSolution);

        // 4. restore ls_.rhs() to the rhs-free state so the following rAU/HbyA read the H-system.
        {
            auto [vol, src, rhsV] = NeoN::views(psi_.mesh().cellVolumes(), expSource, ls_.rhs());
            NeoN::parallelFor(
                psi_.exec(),
                {0, rhsV.size()},
                NEON_LAMBDA(const NeoN::localIdx i) { rhsV[i] += src[i] * vol[i]; }
            );
        }

        return stats;
    }

    // Assemble for a solve honouring the matrix-reuse setting: a full matrix+rhs assemble on the
    // first solve (or when invalidated via markMatrixDirty, or while a setReference pin is active),
    // otherwise an rhs-only refresh that reuses the cached matrix. The rhs-only path exists for the
    // same-type (scalar-matrix == field-type) form only; the segregated vector solve, and any
    // PDESolver with reuse disabled, always re-assembles in full — i.e. behaviour is unchanged
    // unless enableMatrixReuse() was called.
    void assembleForSolve(dsl::Expression<ValueType>& expr, LinearSystem& ls)
    {
        if constexpr (std::is_same_v<MatrixValueType, ValueType>)
        {
            if (matrixReuse_ && matrixAssembled_ && !needReference_)
            {
                ls.resetRhs();
                expr.assembleRhs(ls, psi_.mesh());
                return;
            }
        }
        ls.reset();
        expr.assemble(runTime_.t, runTime_.dt, ls, psi_.mesh());
        matrixAssembled_ = true;
    }

    // Public because NVCC forbids extended __host__ __device__ lambdas
    // (NEON_LAMBDA) inside private or protected member functions.
    NeoN::la::SolverStats solveImpl(dsl::Expression<ValueType>& expr, LinearSystem& ls)
    {
        // Re-read schemes (idempotent with the constructor read)
        expr.read(runTime_.fvSchemesDict);

        // Assemble without post-assembly functors; we apply SetReference separately below
        // to ensure correct polymorphic dispatch — storing PostAssemblyBase by value causes
        // object slicing that silently disables virtual overrides. assembleForSolve honours the
        // matrix-reuse setting (full matrix+rhs assemble vs rhs-only refresh).
        assembleForSolve(expr, ls);

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
        // Drop NeoFOAM-only keys before handing the dict to NeoN/Ginkgo, whose
        // config parser rejects unknown keys (e.g. assemblyStrategy, optimize).
        stripNeoFOAMKeys(fieldSolverDict);
        NeoN::fence(psi_.exec());
        NF_ASSERT(ls.exec() == psi_.exec(), "Executors are not the same");

        auto solver = NeoN::la::Solver(psi_.exec(), fieldSolverDict);
        auto stats = solver.solve(ls, psi_.internalVector());

        reportSolverStats(stats, fieldSolverDict);

        return stats;
    }

private:

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
    // Matrix-reuse cadence: when matrixReuse_ is enabled the assembled matrix is kept across
    // solves and only the rhs is refreshed; matrixAssembled_ tracks whether ls_ already holds a
    // valid matrix (set by a full assemble, cleared by markMatrixDirty / enableMatrixReuse).
    bool matrixReuse_ = false;
    bool matrixAssembled_ = false;
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


template<typename ValueType>
NeoN::finiteVolume::cellCentred::VolumeField<ValueType> operator&(
    const PDESolver<ValueType>& expr,
    const NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& psi
)
{
    return applyOperator(expr.linearSystem(), psi);
}

}
