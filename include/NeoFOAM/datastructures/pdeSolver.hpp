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
 //   using PostBase = NeoN::dsl::PostAssemblyBase<ValueType>;

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
    //struct SetReference final: public NeoN::dsl::PostAssemblyBase<NeoN::scalar>
    {

        NeoN::localIdx pRefCell_;
        NeoN::scalar pRefValue_;

        SetReference(NeoN::localIdx pRefCell, NeoN::scalar pRefValue)
            : pRefCell_(pRefCell)
            , pRefValue_(pRefValue)
        {}

        virtual void operator()(
            const NeoN::la::SparsityPattern& sp,
            NeoN::la::LinearSystem<NeoN::scalar, NeoN::localIdx>& ls
        ) //const override
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
	    //NeoN::Logging::info(
            //    "SetReference applied at cell {}, value {}",
            //    pRefCell_, pRefValue_
            //);
        }
    };

    template<typename Op>
    const Op& temporalOperator() const
    {
        for (const auto& op : expr_.temporalOperators())
        {
            if (const auto* typed = dynamic_cast<const Op*>(&op))
            {
                return *typed;
            }
        }

        NF_ERROR_EXIT("Requested temporal operator not found in expression");
    }

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
        auto ls = NeoN::la::LinearSystem<ValueType, IndexType>(ls_);
        expr.addOperator(-1.0 * rhs);
        assemble();
        return solveImpl(expr, ls);
    }

private:

    NeoN::la::SolverStats
    solveImpl(dsl::Expression<ValueType>& expr, NeoN::la::LinearSystem<ValueType, IndexType>& ls)
    {
        // Only if ValueType is scalar
        //auto functs = std::vector<const NeoN::dsl::PostAssemblyBase<ValueType>*> {};
	auto functs = std::vector<NeoN::dsl::PostAssemblyBase<ValueType>> {};

        if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
        {
	   /* if (ddtScheme_ != nullptr)
            {
		auto& opt = ddtFluxCorrFunctor_;
		opt.emplace(
                    *ddtScheme_,
                    *ddtU_,
                    *ddtFlux_,
                    runTime_.dt
                );
                functs.push_back(&*opt);
                //functs.push_back(
                //    NeoN::dsl::DdtFluxCorr<ValueType>(
                //        *ddtScheme_,
                //        *ddtU_,
                //        *ddtFlux_,
                //        runTime_.dt
                //    )
                //);
            }*/
	    if (needReference_)
            {
		//auto& opt = setReferenceFunctor_;
		//opt.emplace(pRefCell_, pRefValue_);
                //functs.push_back(&*opt);
                functs.push_back(
                    SetReference<ValueType>(pRefCell_, pRefValue_)
                );
            }
        }

        auto solverDict = runTime_.fvSolutionDict.subDict("solvers");
        auto fieldSolverDict = solverDict.subDict(psi_.name);

        auto stats = NeoN::dsl::detail::iterativeSolveImpl(
            expr,
            sparsityPattern_,
            ls,
            psi_,
            runTime_.t,
            runTime_.dt,
            runTime_.fvSchemesDict,
            fieldSolverDict,
	    functs
            //std::span<const NeoN::dsl::PostAssemblyBase<ValueType>* const>{functs.data(), functs.size()}
        );

        NeoN::Logging::info(
            "Solving for {} Initial residual: {} Final residual: {} No Iterations: {}",
                psi_.name, stats.initResNorm, stats.finalResNorm, stats.numIter);
        return stats;
    }


    VolumeField& psi_;
    dsl::Expression<ValueType> expr_;
    const RunTime& runTime_;
    const NeoN::la::SparsityPattern& sparsityPattern_;
    NeoN::la::LinearSystem<ValueType, IndexType> ls_;

    bool needReference_{false};
    NeoN::localIdx pRefCell_{NeoN::localIdx(-1)};
    NeoN::scalar pRefValue_{NeoN::scalar(0)};

    /*using SetRefOpt = std::conditional_t<
        std::is_same_v<ValueType, NeoN::scalar>,
        std::optional<SetReference>,
        std::monostate
    >;

    using DdtFluxOpt = std::conditional_t<
        std::is_same_v<ValueType, NeoN::scalar>,
        std::optional<NeoN::dsl::DdtFluxCorr<NeoN::scalar>>,
        std::monostate
    >;
    SetRefOpt  setReferenceFunctor_{};
    DdtFluxOpt ddtFluxCorrFunctor_{};
    */
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
