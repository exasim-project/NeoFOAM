// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
// TODO: move to cellCenred dsl?

#pragma once

#include "NeoFOAM/linearAlgebra/CSRMatrix.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/linearAlgebra/solver.hpp"
#include "NeoFOAM/dsl/expression.hpp"
#include "NeoFOAM/dsl/solver.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"

namespace dsl = NeoFOAM::dsl;

namespace NeoFOAM::finiteVolume::cellCentred
{

/*@brief extends expression by giving access to assembled matrix
 * @note used in neoIcoFOAM directly instead of dsl::expression
 * TODO: implement flag if matrix is assembled or not -> if not assembled call assemble
 * for dependent operations like discrete momentum fields
 * needs storage for assembled matrix? and whether update is needed like for rAU and HbyA
 */
template<typename ValueType, typename IndexType = localIdx>
class Expression
{
public:

    Expression(
        dsl::Expression<ValueType> expr,
        VolumeField<ValueType>& psi,
        [[maybe_unused]] const Dictionary& fvSchemes,
        [[maybe_unused]] const Dictionary& fvSolution
    )
        : psi_(psi), expr_(expr), fvSchemes_(fvSchemes), fvSolution_(fvSolution),
          sparsityPattern_(SparsityPattern::readOrCreate(psi.mesh())),
          ls_(la::createEmptyLinearSystem<ValueType, localIdx, SparsityPattern>(
              *sparsityPattern_.get()
          ))
    {
        expr_.build(fvSchemes_);
        // assemble();
    };

    Expression(const Expression& ls)
        : psi_(ls.psi_), expr_(ls.expr_), fvSchemes_(ls.fvSchemes_), fvSolution_(ls.fvSolution_),
          ls_(ls.ls_), sparsityPattern_(ls.sparsityPattern_) {};

    ~Expression() = default;

    [[nodiscard]] la::LinearSystem<ValueType, IndexType>& linearSystem() { return ls_; }
    [[nodiscard]] SparsityPattern& sparsityPattern()
    {
        if (!sparsityPattern_)
        {
            NF_THROW(std::string("fvcc:LinearSystem:sparsityPattern: sparsityPattern is null"));
        }
        return *sparsityPattern_;
    }

    VolumeField<ValueType>& getField() { return this->psi_; }

    const VolumeField<ValueType>& getField() const { return this->psi_; }

    [[nodiscard]] const la::LinearSystem<ValueType, IndexType>& linearSystem() const { return ls_; }
    [[nodiscard]] const SparsityPattern& sparsityPattern() const
    {
        if (!sparsityPattern_)
        {
            NF_THROW("fvcc:LinearSystem:sparsityPattern: sparsityPattern is null");
        }
        return *sparsityPattern_;
    }

    const Executor& exec() const { return ls_.exec(); }


    void assemble(scalar t, scalar dt)
    {
        auto vol = psi_.mesh().cellVolumes().view();
        auto expSource = expr_.explicitOperation(psi_.mesh().nCells());
        expr_.explicitOperation(expSource, t, dt);
        auto expSourceView = expSource.view();
        fill(ls_.rhs(), zero<ValueType>());
        fill(ls_.matrix().values(), zero<ValueType>());
        expr_.implicitOperation(ls_);
        // TODO rename implicitOperation -> assembleLinearSystem
        expr_.implicitOperation(ls_, t, dt);
        auto rhs = ls_.rhs().view();
        // we subtract the explicit source term from the rhs
        NeoFOAM::parallelFor(
            exec(),
            {0, rhs.size()},
            KOKKOS_LAMBDA(const size_t i) { rhs[i] -= expSourceView[i] * vol[i]; }
        );
    }

    void assemble()
    {
        if (expr_.temporalOperators().size() == 0 && expr_.spatialOperators().size() == 0)
        {
            NF_ERROR_EXIT("No temporal or implicit terms to solve.");
        }

        if (expr_.temporalOperators().size() > 0)
        {
            // integrate equations in time
            // NeoFOAM::timeIntegration::TimeIntegration<VolumeField<ValueType>> timeIntegrator(
            //     fvSchemes_.subDict("ddtSchemes"), fvSolution_
            // );
            // timeIntegrator.solve(expr_, psi_, t, dt);
        }
        else
        {
            // solve sparse matrix system
            auto vol = psi_.mesh().cellVolumes().view();
            auto expSource = expr_.explicitOperation(psi_.mesh().nCells());
            auto expSourceSpan = expSource.view();

            ls_ = expr_.implicitOperation();
            auto rhs = ls_.rhs().view();
            // we subtract the explicit source term from the rhs
            NeoFOAM::parallelFor(
                exec(),
                {0, rhs.size()},
                KOKKOS_LAMBDA(const size_t i) { rhs[i] -= expSourceSpan[i] * vol[i]; }
            );
        }
    }

    // TODO unify with dsl/solver.hpp
    void solve(scalar t, scalar dt)
    {
        // dsl::solve(expr_, psi_, t, dt, fvSchemes_, fvSolution_);
        if (expr_.temporalOperators().size() == 0 && expr_.spatialOperators().size() == 0)
        {
            NF_ERROR_EXIT("No temporal or implicit terms to solve.");
        }
        if (expr_.temporalOperators().size() > 0)
        {
            NF_ERROR_EXIT("Not implemented");
            //     // integrate equations in time
            //     NeoFOAM::timeIntegration::TimeIntegration<VolumeField<ValueType>> timeIntegrator(
            //         fvSchemes_.subDict("ddtSchemes"), fvSolution_
            //     );
            //     timeIntegrator.solve(expr_, psi_, t, dt);
        }
        else
        {
            auto exec = psi_.exec();
            auto solver = NeoFOAM::la::Solver(exec, fvSolution_);
            solver.solve(ls_, psi_.internalField());
            NF_ERROR_EXIT("No linear solver is available, build with -DNEOFOAM_WITH_GINKGO=ON");
        }
    }

    void setReference(const IndexType refCell, ValueType refValue)
    {
        // TODO currently assumes that matrix is already assembled
        const auto diagOffset = sparsityPattern_->diagOffset().view();
        const auto rowPtrs = ls_.matrix().rowPtrs();
        auto rhs = ls_.rhs().view();
        auto values = ls_.matrix().values().view();
        NeoFOAM::parallelFor(
            ls_.exec(),
            {refCell, refCell + 1},
            KOKKOS_LAMBDA(const std::size_t refCelli) {
                auto diagIdx = rowPtrs[refCelli] + diagOffset[refCelli];
                auto diagValue = values[diagIdx];
                rhs[refCelli] += diagValue * refValue;
                values[diagIdx] += diagValue;
            }
        );
    }

private:

    VolumeField<ValueType>& psi_;
    dsl::Expression<ValueType> expr_;
    const Dictionary& fvSchemes_;
    const Dictionary& fvSolution_;
    std::shared_ptr<SparsityPattern> sparsityPattern_;
    la::LinearSystem<ValueType, IndexType> ls_;
};

template<typename ValueType, typename IndexType = localIdx>
VolumeField<ValueType>
operator&(const Expression<ValueType, IndexType> expr, const VolumeField<ValueType>& psi)
{
    VolumeField<ValueType> resultField(
        psi.exec(),
        "ls_" + psi.name,
        psi.mesh(),
        psi.internalField(),
        psi.boundaryField(),
        psi.boundaryConditions()
    );

    auto [result, b, x] =
        spans(resultField.internalField(), expr.linearSystem().rhs(), psi.internalField());
    const auto [values, colIdxs, rowPtrs] = expr.linearSystem().view();

    NeoFOAM::parallelFor(
        resultField.exec(),
        {0, result.size()},
        KOKKOS_LAMBDA(const std::size_t rowi) {
            IndexType rowStart = rowPtrs[rowi];
            IndexType rowEnd = rowPtrs[rowi + 1];
            ValueType sum = 0.0;
            for (IndexType coli = rowStart; coli < rowEnd; coli++)
            {
                sum += values[coli] * x[colIdxs[coli]];
            }
            result[rowi] = sum - b[rowi];
        }
    );

    return resultField;
}

}
