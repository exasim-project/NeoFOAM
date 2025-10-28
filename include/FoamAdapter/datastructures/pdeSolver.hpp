// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 FoamAdapter authors
// TODO: move to cellCenred dsl?

#pragma once

#include "NeoN/NeoN.hpp"

#include "FoamAdapter/datastructures/runTime.hpp"
#include "FoamAdapter/compatibility/fvSolution.hpp"

namespace dsl = NeoN::dsl;

namespace FoamAdapter
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

public:


    PDESolver(dsl::Expression<ValueType> expr, VolumeField& psi, const RunTime& runTime)
        : psi_(psi)
        , expr_(expr)
        , runTime_(runTime)
        , sparsityPattern_(NeoN::la::SparsityPattern::readOrCreate(psi.mesh()))
        , ls_(NeoN::la::createEmptyLinearSystem<ValueType, NeoN::localIdx>(
              psi.mesh(),
              sparsityPattern_
          ))
    {
        expr_.read(runTime_.fvSchemesDict);
    };

    PDESolver(const PDESolver& expr)
        : psi_(expr.psi_)
        , expr_(expr.expr_)
        , runTime_(expr.runTime_)
        , ls_(expr.ls_)
        , sparsityPattern_(expr.sparsityPattern_) {};

    ~PDESolver() = default;

    VolumeField& getField() { return this->psi_; }

    const VolumeField& getField() const { return this->psi_; }

    [[nodiscard]] const NeoN::la::SparsityPattern& sparsityPattern() const
    {
        return sparsityPattern_;
    }

    [[nodiscard]] NeoN::la::LinearSystem<ValueType, IndexType>& linearSystem() { return ls_; }

    [[nodiscard]] const NeoN::la::LinearSystem<ValueType, IndexType>& linearSystem() const
    {
        return ls_;
    }

    NeoN::la::LinearSystem<ValueType, IndexType>& assemble()
    {
        expr_.assemble(runTime_.t, runTime_.dt, sparsityPattern_, ls_);
        return ls_;
    }

    const NeoN::Executor& exec() const { return ls_.exec(); }


    template<typename FunctorValueType>
    struct SetReference : public NeoN::dsl::PostAssemblyBase<ValueType>
    {

        NeoN::localIdx pRefCell_;
        NeoN::scalar pRefValue_;

        SetReference(NeoN::localIdx pRefCell, NeoN::scalar pRefValue)
            : pRefCell_(pRefCell)
            , pRefValue_(pRefValue)
        {}

        virtual void operator()(
            const NeoN::la::SparsityPattern& sp,
            NeoN::la::LinearSystem<FunctorValueType, NeoN::localIdx>& ls
        )
        {
            const auto diagOffset = sp.diagOffset().view();
            const auto rowOffs = ls.matrix().rowOffs().view();
            auto rhs = ls.rhs().view();
            auto values = ls.matrix().values().view();
            // make an explicit copy to avoid capture this warning in kokkos lambda
            auto pRefValue = pRefValue_;

            NeoN::parallelFor(
                ls.exec(),
                {pRefCell_, pRefCell_ + 1},
                KOKKOS_LAMBDA(const std::size_t refCelli) {
                    auto diagIdx = rowOffs[refCelli] + diagOffset[refCelli];
                    auto diagValue = values[diagIdx];
                    rhs[refCelli] += diagValue * pRefValue;
                    values[diagIdx] += diagValue;
                }
            );
        }
    };

    void setReference(NeoN::localIdx pRefCell, NeoN::scalar pRefValue)
    {
        needReference_ = true;
        pRefCell_ = pRefCell;
        pRefValue_ = pRefValue;
    }

    // TODO unify with dsl/solver.hpp
    NeoN::la::SolverStats solve(
            const dsl::Expression<ValueType>& expr,
            const NeoN::la::LinearSystem<ValueType, IndexType>& ls)
    {
        // Only if ValueType is scalar
        auto functs = std::vector<NeoN::dsl::PostAssemblyBase<ValueType>> {};

        if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
        {
            functs =
                needReference_
                    ? std::vector<NeoN::dsl::PostAssemblyBase<ValueType>> {SetReference<ValueType>(
                          pRefCell_,
                          pRefValue_
                      )}
                    : std::vector<NeoN::dsl::PostAssemblyBase<ValueType>> {};
        }

        // FIXME TODO this will create the sparsity pattern and potentially the ls
        // again even if it has been created already
        auto solverDict = runTime_.fvSolutionDict.get<NeoN::Dictionary>("solvers");
        auto fieldSolverDict = solverDict.get<NeoN::Dictionary>(psi_.name);
        auto stats = NeoN::dsl::detail::iterativeSolveImpl(
            expr_,
            sparsityPattern_,
            ls_,
            psi_,
            runTime_.t,
            runTime_.dt,
            runTime_.fvSchemesDict,
            fieldSolverDict,
            functs
        );

        std::cout << "[NeoN] Solving for " << psi_.name << ":"
                  << " Initial residual: " << stats.initResNorm
                  << " Final residual: " << stats.finalResNorm
                  << " No Iterations: " << stats.numIter << std::endl;
        return stats;
    }

    NeoN::la::SolverStats solve() {
        return solve(expr_, ls_);
    }

    NeoN::la::SolverStats solve(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
    {
        auto expr = dsl::Expression<ValueType>(expr_);
        auto ls = NeoN::la::LinearSystem<ValueType, IndexType>(ls_);
        expr.addOperator(-1.0 * rhs);
        return solve(expr, ls);
    }

private:

    VolumeField& psi_;
    dsl::Expression<ValueType> expr_;
    const RunTime& runTime_;
    const NeoN::la::SparsityPattern& sparsityPattern_;
    NeoN::la::LinearSystem<ValueType, IndexType> ls_;

    bool needReference_;
    NeoN::localIdx pRefCell_;
    NeoN::scalar pRefValue_;
};

template<typename ValueType, typename IndexType = NeoN::localIdx>
NeoN::Vector<ValueType> diag(
    const la::LinearSystem<ValueType, IndexType>& ls,
    const NeoN::la::SparsityPattern& sparsityPattern
)
{
    NeoN::Vector<ValueType> diagonal(ls.exec(), sparsityPattern.diagOffset().size(), 0.0);
    auto diagView = diagonal.view();

    const auto diagOffset = sparsityPattern.diagOffset().view();
    const auto [matrix, b] = ls.view();
    NeoN::parallelFor(
        ls.exec(),
        {0, diagOffset.size()},
        KOKKOS_LAMBDA(const std::size_t celli) {
            auto diagOffsetCelli = diagOffset[celli];
            diagView[celli] = matrix.values[matrix.rowOffs[celli] + diagOffsetCelli];
        }
    );
    return diagonal;
}


template<typename ValueType, typename IndexType = NeoN::localIdx>
NeoN::finiteVolume::cellCentred::VolumeField<ValueType> applyOperator(
    const la::LinearSystem<ValueType, IndexType>& ls,
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
