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
template<typename ValueType>
class PDESolver
{
    using VolumeField = NeoN::finiteVolume::cellCentred::VolumeField<ValueType>;
    using LinearSystem = NeoN::la::LinearSystem<ValueType>;

public:


    PDESolver(dsl::Expression<ValueType> expr, VolumeField& psi, const RunTime& runTime)
        : psi_(psi)
        , expr_(expr)
        , runTime_(runTime)
        , ls_(NeoN::la::createEmptyLinearSystem<ValueType>(psi.mesh()))
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

    LinearSystem& assemble()
    {
        ls_.reset();
        expr_.assemble(runTime_.t, runTime_.dt, ls_);
        NeoN::dsl::detail::addBoundaryContributions(ls_);
        return ls_;
    }

    const NeoN::Executor& exec() const { return ls_.exec(); }


    template<typename FunctorValueType>
    struct SetReference : public NeoN::dsl::PostAssemblyBase<FunctorValueType, NeoN::localIdx>
    {

        NeoN::localIdx pRefCell_;
        NeoN::scalar pRefValue_;

        SetReference(NeoN::localIdx pRefCell, NeoN::scalar pRefValue)
            : pRefCell_(pRefCell)
            , pRefValue_(pRefValue)
        {}

        virtual void operator()(
            NeoN::la::LinearSystem<FunctorValueType, NeoN::la::CSRMatrix<FunctorValueType, NeoN::localIdx>>& ls
        ) override
        {
            const auto diagOffset = ls.faceToMatrixAddress()->diagOffset().view();
            const auto rowOffs = ls.matrix().rowOffs().view();
            auto rhs = ls.rhs().view();
            auto values = ls.matrix().values().view();
            // make an explicit copy to avoid capture this warning in kokkos lambda
            auto pRefValue = pRefValue_;

            NeoN::parallelFor(
                ls.exec(),
                {pRefCell_, pRefCell_ + 1},
                NEON_LAMBDA(const std::size_t refCelli) {
                    auto diagIdx = rowOffs[refCelli] + diagOffset[refCelli];
                    auto diagValue = values[diagIdx];
                    rhs[refCelli] += diagValue * pRefValue;
                    values[diagIdx] += diagValue;
                }
            );
        }
    };

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
        needReference_ = true;
        pRefCell_ = pRefCell;
        pRefValue_ = pRefValue;
    }

    NeoN::la::SolverStats solve() { return solveImpl(expr_, ls_); }

    NeoN::la::SolverStats solve(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
    {
        auto expr = dsl::Expression<ValueType>(expr_);
        auto ls = LinearSystem(ls_);
        expr.addOperator(-1.0 * rhs);
        assemble();
        return solveImpl(expr, ls);
    }

private:

    NeoN::la::SolverStats
    solveImpl(dsl::Expression<ValueType>& expr, LinearSystem& ls)
    {
        // Only if ValueType is scalar
        auto functs = std::vector<NeoN::dsl::PostAssemblyBase<ValueType, NeoN::localIdx>> {};

        if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
        {
            functs =
                needReference_
                    ? std::vector<NeoN::dsl::PostAssemblyBase<ValueType, NeoN::localIdx>> {SetReference<ValueType>(
                        pRefCell_,
                        pRefValue_
                    )}
                    : std::vector<NeoN::dsl::PostAssemblyBase<ValueType, NeoN::localIdx>> {};
        }
	// Resolve URFs once per solve call (CPU-side)
        const auto eqnUrf =
            NeoN::dsl::detail::findRelaxationFactor(runTime_.fvSolutionDict, psi_.name);

        const auto fieldUrf =
            NeoN::dsl::detail::findFieldRelaxationFactor(runTime_.fvSolutionDict, psi_.name);

        if (eqnUrf)
        {
            NeoN::Logging::info("URF equation {} = {}", psi_.name, *eqnUrf);
        }
        if (fieldUrf)
        {
            NeoN::Logging::info("URF field {} = {}", psi_.name, *fieldUrf);
        }

        auto solverDict = runTime_.fvSolutionDict.subDict("solvers");
        auto fieldSolverDict = solverDict.subDict(psi_.name);

        auto stats = NeoN::dsl::detail::iterativeSolveImpl(
            expr,
            ls,
            psi_,
            runTime_.t,
            runTime_.dt,
            runTime_.fvSchemesDict,
            fieldSolverDict,
            functs,
	    eqnUrf,
	    fieldUrf
        );

        if (!stats.entries.empty())
        {
            const auto& e = stats.entries.back();
            NeoN::Logging::info(
                "Solving for {} Initial residual: {} Final residual: {} No Iterations: {}",
                    psi_.name, e.initResNorm, e.finalResNorm, e.numIter);
        }
        return stats;
    }


    VolumeField& psi_;
    dsl::Expression<ValueType> expr_;
    const RunTime& runTime_;
    LinearSystem ls_;

    bool needReference_ = false;
    NeoN::localIdx pRefCell_ = 0;
    NeoN::scalar pRefValue_ = 0.0;
};

template<typename ValueType>
NeoN::Vector<ValueType> diag(const NeoN::la::LinearSystem<ValueType>& ls)
{
    const auto matIt = ls.faceToMatrixAddress();
    NeoN::Vector<ValueType> diagonal(ls.exec(), matIt->diagOffset().size(), 0.0);
    auto diagView = diagonal.view();

    const auto diagOffset = matIt->diagOffset().view();
    const auto [matrix, rhs, bMatrix, bRhs] = ls.view();
    NeoN::parallelFor(
        ls.exec(),
        {0, diagOffset.size()},
        NEON_LAMBDA(const std::size_t celli) {
            auto diagOffsetCelli = diagOffset[celli];
            diagView[celli] = matrix.values[matrix.rowOffs[celli] + diagOffsetCelli];
        }
    );
    return diagonal;
}


template<typename ValueType>
NeoN::finiteVolume::cellCentred::VolumeField<ValueType> applyOperator(
    const NeoN::la::LinearSystem<ValueType>& ls,
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
    const PDESolver<ValueType> expr,
    const NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& psi
)
{
    return applyOperator(expr.linearSystem(), psi);
}

}
