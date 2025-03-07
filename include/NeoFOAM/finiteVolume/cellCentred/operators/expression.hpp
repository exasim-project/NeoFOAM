// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoFOAM/linearAlgebra/CSRMatrix.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/linearAlgebra/ginkgo.hpp"
#include "NeoFOAM/linearAlgebra/petscSolver.hpp"
#include "NeoFOAM/dsl/expression.hpp"
#include "NeoFOAM/dsl/solver.hpp"


#include "NeoFOAM/finiteVolume/cellCentred/operators/sparsityPattern.hpp"

namespace dsl = NeoFOAM::dsl;

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType, typename IndexType = localIdx>
class Expression
{
public:

    Expression(
        dsl::Expression expr,
        VolumeField<ValueType>& psi,
        [[maybe_unused]] const Dictionary& fvSchemes,
        [[maybe_unused]] const Dictionary& fvSolution
    )
        : psi_(psi), expr_(expr), fvSchemes_(fvSchemes), fvSolution_(fvSolution),
          ls_(SparsityPattern::readOrCreate(psi.mesh())->linearSystem()),
          sparsityPattern_(SparsityPattern::readOrCreate(psi.mesh()))
    {
        expr_.build(fvSchemes_);
        assemble();
    };

    Expression(const Expression& ls)
        : psi_(ls.psi_), expr_(ls.expr_), fvSchemes_(ls.fvSchemes), fvSolution_(ls.fvSolution),
          ls_(ls.ls_), sparsityPattern_(ls.sparsityPattern_) {};

    ~Expression() = default;

    [[nodiscard]] la::LinearSystem<ValueType, IndexType>& linearSystem() { return ls_; }
    [[nodiscard]] SparsityPattern& sparsityPattern()
    {
        if (!sparsityPattern_)
        {
            NF_THROW("fvcc:LinearSystem:sparsityPattern: sparsityPattern is null");
        }
        return *sparsityPattern_;
    }

    VolumeField<ValueType>& getField() { return psi_; }

    const VolumeField<ValueType>&& getField() const { return psi_; }

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

    void diag(Field<ValueType>& field)
    {
        NF_ASSERT_EQUAL(field.size(), sparsityPattern_->diagOffset().size());
        const auto diagOffset = sparsityPattern_->diagOffset().span();
        const auto rowPtrs = ls_.matrix().rowPtrs();
        std::span<ValueType> fieldSpan = field.span();
        std::span<ValueType> values = ls_.matrix().values();
        NeoFOAM::parallelFor(
            exec(),
            {0, diagOffset.size()},
            KOKKOS_LAMBDA(const std::size_t celli) {
                auto diagOffsetCelli = diagOffset[celli];
                fieldSpan[celli] = values[rowPtrs[celli] + diagOffsetCelli];
            }
        );
    }

    Field<ValueType> diag()
    {
        Field<ValueType> diagonal(exec(), sparsityPattern_->diagOffset().size(), 0.0);
        diag(diagonal);
        return diagonal;
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
            auto vol = psi_.mesh().cellVolumes().span();
            auto expSource = expr_.explicitOperation(psi_.mesh().nCells());
            auto expSourceSpan = expSource.span();

            ls_ = expr_.implicitOperation();
            auto rhs = ls_.rhs().span();
            // we subtract the explicit source term from the rhs
            NeoFOAM::parallelFor(
                exec(),
                {0, rhs.size()},
                KOKKOS_LAMBDA(const size_t i) { rhs[i] -= expSourceSpan[i] * vol[i]; }
            );
        }
    }

    void solve(scalar t, scalar dt)
    {
        // dsl::solve(expr_, psi_, t, dt, fvSchemes_, fvSolution_);
        // FIXME:
        if (expr_.temporalOperators().size() == 0 && expr_.spatialOperators().size() == 0)
        {
            NF_ERROR_EXIT("No temporal or implicit terms to solve.");
        }
        if (expr_.temporalOperators().size() > 0)
        {
            //     // integrate equations in time
            //     NeoFOAM::timeIntegration::TimeIntegration<VolumeField<ValueType>> timeIntegrator(
            //         fvSchemes_.subDict("ddtSchemes"), fvSolution_
            //     );
            //     timeIntegrator.solve(expr_, psi_, t, dt);
        }
        else
        {
#if NF_WITH_GINKGO
            NeoFOAM::la::ginkgo::BiCGStab<ValueType> solver(psi_.exec(), fvSolution_);
            auto convertedLS = convertLinearSystem(ls_);
            solver.solve(convertedLS, psi_.internalField());
#elif NF_WITH_PETSC
            // placeholder for petsc solver --> include runtime selection in future
            // should look like this
            NeoFOAM::la::petscSolver::petscSolver<ValueType> solver(psi_.exec(), fvSolution_);
            solver.solve(ls_, psi_.internalField());
#endif
        }
    }

    Field<IndexType> diagIndex()
    {
        Field<IndexType> diagIndex(exec(), sparsityPattern_->diagOffset().size());
        const auto diagOffset = sparsityPattern_->diagOffset().span();
        auto diagIndexSpan = diagIndex.span();
        const auto rowPtrs = ls_.matrix().rowPtrs();
        NeoFOAM::parallelFor(
            exec(),
            {0, diagIndex.size()},
            KOKKOS_LAMBDA(const std::size_t celli) {
                diagIndexSpan[celli] = rowPtrs[celli] + diagOffset[celli];
            }
        );
        return diagIndex;
    }

    void setReference(const IndexType refCell, ValueType refValue)
    {
        // TODO currently assumes that matrix is already assembled
        const auto diagOffset = sparsityPattern_->diagOffset().span();
        const auto rowPtrs = ls_.matrix().rowPtrs();
        auto rhs = ls_.rhs().span();
        auto values = ls_.matrix().values();
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

    Field<ValueType> flux() const
    {
        const UnstructuredMesh& mesh = psi_.mesh();
        const std::size_t nInternalFaces = mesh.nInternalFaces();
        const auto exec = psi_.exec();
        const auto [owner, neighbour, surfFaceCells, ownOffs, neiOffs, internalPsi] = spans(
            mesh.faceOwner(),
            mesh.faceNeighbour(),
            mesh.boundaryMesh().faceCells(),
            sparsityPattern_->ownerOffset(),
            sparsityPattern_->neighbourOffset(),
            psi_.internalField()
        );

        const auto rowPtrs = ls_.matrix().rowPtrs();
        const auto colIdxs = ls_.matrix().colIdxs();
        auto values = ls_.matrix().values();
        auto rhs = ls_.rhs().span();


        Field<ValueType> result(exec, neighbour.size(), 0.0);

        auto resultSpan = result.span();

        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t facei) {
                std::size_t own = static_cast<std::size_t>(owner[facei]);
                std::size_t nei = static_cast<std::size_t>(neighbour[facei]);

                std::size_t rowNeiStart = rowPtrs[nei];
                std::size_t rowOwnStart = rowPtrs[own];

                auto Upper = values[rowNeiStart + neiOffs[facei]];
                auto Lower = values[rowOwnStart + ownOffs[facei]];
                Kokkos::atomic_add(
                    &resultSpan[facei], Upper * internalPsi[nei] - Lower * internalPsi[own]
                );
            }
        );
        return result;
    }

private:

    VolumeField<ValueType>& psi_;
    dsl::Expression expr_;
    const Dictionary& fvSchemes_;
    const Dictionary& fvSolution_;
    la::LinearSystem<ValueType, IndexType> ls_;
    std::shared_ptr<SparsityPattern> sparsityPattern_;
};

template<typename ValueType, typename IndexType = localIdx>
VolumeField<ValueType>
operator&(const Expression<ValueType, IndexType> ls, const VolumeField<ValueType>& psi)
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
        spans(resultField.internalField(), ls.linearSystem().rhs(), psi.internalField());

    const std::span<const ValueType> values = ls.linearSystem().matrix().values();
    const std::span<const IndexType> colIdxs = ls.linearSystem().matrix().colIdxs();
    const std::span<const IndexType> rowPtrs = ls.linearSystem().matrix().rowPtrs();

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
