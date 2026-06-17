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

    /** @brief Select the final-iteration relaxation factor (the <field>Final key).
     *
     * Defaults to false (base <field> key). The PIMPLE driver sets it from
     * pimpleControl.finalIter() so the final outer corrector uses the *Final factor.
     */
    void setFinalIter(bool finalIter) { finalIter_ = finalIter; }

    /** @brief assemble the linear system owned by the solver based on the current expression */
    LinearSystem& assemble()
    {
        ls_.reset();
        expr_.assemble(runTime_.t, runTime_.dt, ls_, psi_.mesh());
        return ls_;
    }

    /** @brief assemble the OWNED ls_ and relax it in place (no rhs term, no solve).
     *
     * OF-parity: mirrors the assemble()+relax() prefix of solve(rhs) so the owned
     * ls_ carries the RELAXED augmented diagonal that computeRAUandHByA reads (the rAU/HbyA
     * read invariant). OpenFOAM applies UEqn.relax() UNCONDITIONALLY; this lets the
     * `momentumPredictor no` path leave ls_ relaxed exactly as the predictor path does, instead
     * of leaving a bare (un-relaxed) assemble(). The lookup+relax is shared verbatim with
     * solve(rhs) via relaxOwnedLs(), so the two paths cannot drift. alpha==1 (no U-URF) is a
     * bitwise no-op (NeoN kernel early-returns), so neoIcoFoam/neoPisoFoam semantics are
     * unchanged.
     */
    LinearSystem& assembleAndRelax()
    {
        assemble();
        relaxOwnedLs();
        return ls_;
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
        rhsExpr.read(runTime_.fvSchemesDict);
        auto ls = LinearSystem(assemble());
        rhsExpr.assembleExplicitSource(ls, psi_.mesh());

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
        // Read-path fix: relax the OWNED ls_ IN PLACE so computeRAUandHByA — which reads
        // ls_ (pressureVelocityCoupling.cpp) — sees the RELAXED augmented diagonal, mirroring
        // OpenFOAM's UEqn.relax() boosting A() ~1/alpha in place. The system actually SOLVED stays
        // identical to today's (relaxed-diag + relax-source + (-grad p)) to machine precision:
        // today adds -grad p to the copy THEN relaxes the copy; here ls_ is relaxed BEFORE the
        // copy+add. Matrix-URF touches only diag + relax-source (not the explicit -grad p rhs
        // source), so nfU is unchanged → test_momentum stays green. The ONLY new behavior: ls_ now
        // carries the relaxed diagonal for the rAU/HbyA read.

        // 1) Assemble the OWNED ls_ (raw momentum predictor, no rhs term yet) and relax it in
        // place. The lookup+relax is factored into relaxOwnedLs() so the no-predictor path
        // (assembleAndRelax()) shares the SAME alpha lookup and relaxation and the two
        // cannot drift. A missing factor -> 1.0 -> applyMatrixRelaxation early-returns (bitwise
        // no-op for cases without U relax, i.e. neoIcoFoam/neoPisoFoam). Relaxation runs on the
        // RAW assembled diagonal (relax before setReference), exactly as solveImpl.
        assemble();
        relaxOwnedLs();

        // 2) (perf) Apply -grad p to the OWNED ls_'s rhs IN PLACE for
        // the solve instead of DEEP-COPYING the whole momentum LinearSystem. The matrix is the
        // heavy part (~nnz Vec3 entries, ~1 GB at 6.3M cells) and is NOT modified by adding the
        // pressure gradient — assembleExplicitSource touches rhs only, and Solver::solve takes the
        // system as const — so solving ls_ here is byte-identical to solving the old copy. Only the
        // rhs differs (O(nCells)), so we snapshot the rhs-free relaxed-momentum ("H") rhs, apply
        // the source, solve ls_, then RESTORE the snapshot. After restore, ls_ holds exactly what
        // it did before (relaxed diagonal + bare momentum rhs, no -grad p) so the subsequent
        // computeRAUandHByA reads the H-system (the rAU/HbyA read invariant). ls_ is relaxed once
        // (step 1); not re-relaxed.
        auto rhsExpr = dsl::Expression<ValueType>(-1.0 * rhs);
        rhsExpr.read(runTime_.fvSchemesDict);
        auto savedRhs = NeoN::Vector<ValueType>(ls_.rhs());
        rhsExpr.assembleExplicitSource(ls_, psi_.mesh());

        // 3) select the <field>Final solver subdict on the final outer pass;
        // isDict-guarded fallback to the base subdict, identical to solveImpl.
        auto solverDict = runTime_.fvSolutionDict.subDict("solvers");
        const std::string finalKey = psi_.name + "Final";
        auto fvSolution = (finalIter_ && solverDict.isDict(finalKey))
                            ? solverDict.subDict(finalKey)
                            : solverDict.subDict(psi_.name);
        // Drop NeoFOAM-only keys before handing the dict to NeoN/Ginkgo, whose
        // config parser rejects unknown keys (e.g. assemblyStrategy, optimize).
        stripNeoFOAMKeys(fvSolution);
        auto solver = NeoN::la::Solver(psi_.exec(), fvSolution);
        // Do some sanity checks before trying to solve
        // NF_ASSERT(ls.exec() == solution.exec(), "Executors are not the same");
        auto stats = solver.solve(ls_, psi_.internalVector());

        // Restore the rhs-free momentum rhs so computeRAUandHByA reads the H-system, not H - grad
        // p.
        ls_.rhs() = savedRhs;

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
        expr.assemble(runTime_.t, runTime_.dt, ls, psi_.mesh());

        // Equation (matrix) under-relaxation, applied directly here mirroring the
        // SetReference direct call, NOT via a PostAssemblyBase ps vector. The
        // factor is parsed from fvSolution in NeoFOAM; NeoN only sees a scalar.
        // alpha is a plain scalar, so this runs for both scalar and Vec3 fields and lives
        // OUTSIDE the SetReference `if constexpr` guard. A missing entry -> 1.0 -> the
        // kernel early-returns (bitwise no-op). psi_ supplies psi_prev (internalVector)
        // AND the mesh (faceOwners() for the boundary-diagonal owner).
        //
        // Relaxation MUST run on the raw assembled diagonal, BEFORE SetReference
        // (OpenFOAM order: fvMatrix::relax() precedes setReference). SetReference doubles the
        // ref cell's diagonal and adds diag*refVal to rhs; relaxing afterwards would
        // reconstruct/clamp the already-doubled diagonal and corrupt the pin if a field ever
        // carried both a reference cell and a relaxation factor.
        const auto alpha =
            lookupEqnRelaxation(runTime_.fvSolutionDict, psi_.name, finalIter_).value_or(1.0);
        NeoN::dsl::applyMatrixRelaxation(ls, psi_, alpha);

        // Apply reference-cell pinning directly (avoids object-slicing issue)
        if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
        {
            if (needReference_)
            {
                NeoN::dsl::SetReference<ValueType> refFunct(pRefCell_, pRefValue_);
                refFunct(ls);
            }
        }

        // select the <field>Final solver subdict on the final outer pass (tighter
        // pimpleFoam final-pass tolerances); isDict-guarded fallback to the base
        // subdict (a missing/non-dict <field>Final degrades to the base entry instead
        // of throwing out of subDict).
        auto solverDict = runTime_.fvSolutionDict.subDict("solvers");
        const std::string finalKey = psi_.name + "Final";
        auto fieldSolverDict = (finalIter_ && solverDict.isDict(finalKey))
                                 ? solverDict.subDict(finalKey)
                                 : solverDict.subDict(psi_.name);
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

    // Relax the OWNED ls_ in place using the production equation-URF lookup
    // (relaxationFactors.equations.<field>[Final]). Shared verbatim by solve(rhs)
    // and assembleAndRelax() so the predictor and no-predictor paths cannot diverge.
    // alpha defaults to 1.0 (missing factor) -> applyMatrixRelaxation early-returns: a bitwise
    // no-op preserving neoIcoFoam/neoPisoFoam semantics. Mirrors UEqn.relax() and MUST run on
    // the raw assembled diagonal BEFORE any SetReference. Only CALLS the free
    // function applyMatrixRelaxation (which owns the NEON_LAMBDA) — no extended lambda is defined
    // in this member body, so the NVCC private-member-lambda restriction does not apply.
    void relaxOwnedLs()
    {
        const auto alpha =
            lookupEqnRelaxation(runTime_.fvSolutionDict, psi_.name, finalIter_).value_or(1.0);
        NeoN::dsl::applyMatrixRelaxation(ls_, psi_, alpha);
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
    // finalIter seam: selects the <field>Final relaxation key when true. Defaults to false
    // (base key); the PIMPLE driver drives it from pimpleControl.finalIter().
    bool finalIter_ = false;
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
