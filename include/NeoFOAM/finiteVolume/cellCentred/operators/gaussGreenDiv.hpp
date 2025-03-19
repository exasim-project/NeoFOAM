// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/sparsityPattern.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


/* @brief free standing function implementation of the divergence operator
** ie computes 1/V \sum_f S_f \cdot \phi_f
** where S_f is the face normal flux of a given face
**  phi_f is the face interpolate value
**
**
** @param faceFlux
** @param neighbour - mapping from face id to neighbour cell id
** @param owner - mapping from face id to owner cell id
** @param faceCells - mapping from boundary face id to owner cell id
*/
template<typename ValueType>
void computeDiv(
    const Executor& exec,
    size_t nInternalFaces,
    size_t nBoundaryFaces,
    std::span<const int> neighbour,
    std::span<const int> owner,
    std::span<const int> faceCells,
    std::span<const scalar> faceFlux,
    std::span<const ValueType> phiF,
    std::span<const scalar> V,
    std::span<ValueType> res,
    const dsl::Coeff operatorScaling
)
{
    size_t nCells {V.size()};
    // check if the executor is GPU
    if (std::holds_alternative<SerialExecutor>(exec))
    {
        for (size_t i = 0; i < nInternalFaces; i++)
        {
            ValueType flux = faceFlux[i] * phiF[i];
            res[static_cast<size_t>(owner[i])] += flux;
            res[static_cast<size_t>(neighbour[i])] -= flux;
        }

        for (size_t i = nInternalFaces; i < nInternalFaces + nBoundaryFaces; i++)
        {
            auto own = static_cast<size_t>(faceCells[i - nInternalFaces]);
            ValueType valueOwn = faceFlux[i] * phiF[i];
            res[own] += valueOwn;
        }

        // TODO does it make sense to store invVol and multiply?
        for (size_t celli = 0; celli < nCells; celli++)
        {
            res[celli] *= operatorScaling[celli] / V[celli];
        }
    }
    else
    {
        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t i) {
                ValueType flux = faceFlux[i] * phiF[i];
                Kokkos::atomic_add(&res[static_cast<size_t>(owner[i])], flux);
                Kokkos::atomic_sub(&res[static_cast<size_t>(neighbour[i])], flux);
            },
            "sumFluxesInternal"
        );

        parallelFor(
            exec,
            {nInternalFaces, nInternalFaces + nBoundaryFaces},
            KOKKOS_LAMBDA(const size_t i) {
                auto own = static_cast<size_t>(faceCells[i - nInternalFaces]);
                ValueType valueOwn = faceFlux[i] * phiF[i];
                Kokkos::atomic_add(&res[own], valueOwn);
            },
            "sumFluxesInternal"
        );

        parallelFor(
            exec,
            {0, nCells},
            KOKKOS_LAMBDA(const size_t celli) { res[celli] *= operatorScaling[celli] / V[celli]; },
            "normalizeFluxes"
        );
    }
}


template<typename ValueType>
void computeDiv(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceInterpolation<ValueType>& surfInterp,
    Field<ValueType>& divPhi,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    SurfaceField<ValueType> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh)
    );
    // TODO: remove or implement
    // fill(phif.internalField(), NeoFOAM::zero<ValueType>::value);
    surfInterp.interpolate(faceFlux, phi, phif);

    // TODO: currently we just copy the boundary values over
    phif.boundaryField().value() = phi.boundaryField().value();

    size_t nInternalFaces = mesh.nInternalFaces();
    size_t nBoundaryFaces = mesh.nBoundaryFaces();
    computeDiv<ValueType>(
        exec,
        nInternalFaces,
        nBoundaryFaces,
        mesh.faceNeighbour().span(),
        mesh.faceOwner().span(),
        mesh.boundaryMesh().faceCells().span(),
        faceFlux.internalField().span(),
        phif.internalField().span(),
        mesh.cellVolumes().span(),
        divPhi.span(),
        operatorScaling

    );
}

template<typename ValueType>
class GaussGreenDiv :
    public DivOperatorFactory<ValueType>::template Register<GaussGreenDiv<ValueType>>
{
    using Base = DivOperatorFactory<ValueType>::template Register<GaussGreenDiv<ValueType>>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Divergence"; }

    static std::string schema() { return "none"; }

    GaussGreenDiv(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), surfaceInterpolation_(exec, mesh, inputs),
          sparsityPattern_(SparsityPattern::readOrCreate(mesh)) {};

    la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const override
    {
        la::LinearSystem<scalar, localIdx> ls(sparsityPattern_->linearSystem());
        auto [A, b] = ls.view();
        const auto& exec = ls.exec();

        Field<ValueType> values(exec, A.value.size(), zero<ValueType>());
        Field<localIdx> mColIdxs(exec, A.columnIndex.data(), A.columnIndex.size());
        Field<localIdx> mRowPtrs(exec, A.rowOffset.data(), A.rowOffset.size());

        la::CSRMatrix<ValueType, localIdx> matrix(values, mColIdxs, mRowPtrs);
        Field<ValueType> rhs(exec, b.size(), zero<ValueType>());

        return {matrix, rhs, ls.sparsityPattern()};
    };

    virtual void
    div(VolumeField<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) override
    {
        computeDiv<ValueType>(
            faceFlux, phi, surfaceInterpolation_, divPhi.internalField(), operatorScaling
        );
    }

    virtual void
    div(la::LinearSystem<ValueType, localIdx>& ls,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) override
    {
        const UnstructuredMesh& mesh = phi.mesh();
        const std::size_t nInternalFaces = mesh.nInternalFaces();
        const auto exec = phi.exec();
        const auto [sFaceFlux, owner, neighbour, diagOffs, ownOffs, neiOffs] = spans(
            faceFlux.internalField(),
            mesh.faceOwner(),
            mesh.faceNeighbour(),
            sparsityPattern_->diagOffset(),
            sparsityPattern_->ownerOffset(),
            sparsityPattern_->neighbourOffset()
        );
        auto [A, b] = ls.view();

        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t facei) {
                scalar flux = sFaceFlux[facei];
                // scalar weight = 0.5;
                scalar weight = flux >= 0 ? 1 : 0;
                ValueType value = zero<ValueType>();
                std::size_t own = static_cast<std::size_t>(owner[facei]);
                std::size_t nei = static_cast<std::size_t>(neighbour[facei]);

                // add neighbour contribution upper
                std::size_t rowNeiStart = A.rowOffset[nei];
                std::size_t rowOwnStart = A.rowOffset[own];

                value = -weight * flux * one<ValueType>();
                // scalar valueNei = (1 - weight) * flux;
                A.value[rowNeiStart + neiOffs[facei]] += value;
                Kokkos::atomic_sub(&A.value[rowOwnStart + diagOffs[own]], value);

                // upper triangular part
                // add owner contribution lower
                value = flux * (1 - weight) * one<ValueType>();
                A.value[rowOwnStart + ownOffs[facei]] += value;
                Kokkos::atomic_sub(&A.value[rowNeiStart + diagOffs[nei]], value);
            }
        );

        parallelFor(
            exec,
            {0, b.size()},
            KOKKOS_LAMBDA(const size_t celli) {
                b[celli] *= operatorScaling[celli];
                for (size_t i = A.rowOffset[celli]; i < A.rowOffset[celli + 1]; i++)
                {
                    A.value[i] *= operatorScaling[celli];
                }
            }
        );
    };

    virtual void
    div(Field<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) override
    {
        computeDiv<ValueType>(faceFlux, phi, surfaceInterpolation_, divPhi, operatorScaling);
    };

    virtual VolumeField<ValueType>
    div(const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) override
    {
        std::string name = "div(" + faceFlux.name + "," + phi.name + ")";
        VolumeField<ValueType> divPhi(
            this->exec_,
            name,
            this->mesh_,
            createCalculatedBCs<VolumeBoundary<ValueType>>(this->mesh_)
        );
        computeDiv<ValueType>(
            faceFlux, phi, surfaceInterpolation_, divPhi.internalField(), operatorScaling
        );
        return divPhi;
    };

    std::unique_ptr<DivOperatorFactory<ValueType>> clone() const override
    {
        return std::make_unique<GaussGreenDiv<ValueType>>(*this);
    }

private:

    SurfaceInterpolation<ValueType> surfaceInterpolation_;
    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};

} // namespace NeoFOAM
