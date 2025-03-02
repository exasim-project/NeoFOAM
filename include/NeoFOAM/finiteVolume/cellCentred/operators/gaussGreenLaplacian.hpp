// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/laplacianOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/sparsityPattern.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

void computeLaplacian(
    const FaceNormalGradient& faceNormalGradient,
    const SurfaceField<scalar>& gamma,
    VolumeField<scalar>& phi,
    Field<scalar>& lapPhi,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();

    SurfaceField<scalar> faceNormalGrad = faceNormalGradient.faceNormalGrad(phi);

    const auto [owner, neighbour, surfFaceCells] =
        spans(mesh.faceOwner(), mesh.faceNeighbour(), mesh.boundaryMesh().faceCells());

    auto [refGradient, value, valueFraction, refValue] = spans(
        phi.boundaryField().refGrad(),
        phi.boundaryField().value(),
        phi.boundaryField().valueFraction(),
        phi.boundaryField().refValue()
    );

    const auto [result, faceArea, fnGrad, vol] =
        spans(lapPhi, mesh.magFaceAreas(), faceNormalGrad.internalField(), mesh.cellVolumes());


    size_t nInternalFaces = mesh.nInternalFaces();

    // check if the executor is GPU
    if (std::holds_alternative<SerialExecutor>(exec))
    {
        for (size_t i = 0; i < nInternalFaces; i++)
        {
            scalar flux = faceArea[i] * fnGrad[i];
            result[static_cast<size_t>(owner[i])] += flux;
            result[static_cast<size_t>(neighbour[i])] -= flux;
        }

        for (size_t i = nInternalFaces; i < fnGrad.size(); i++)
        {
            auto own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
            scalar valueOwn = faceArea[i] * fnGrad[i];
            result[own] += valueOwn;
        }

        for (size_t celli = 0; celli < mesh.nCells(); celli++)
        {
            result[celli] *= operatorScaling[celli] / vol[celli];
        }
    }
    else
    {
        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t i) {
                scalar flux = faceArea[i] * fnGrad[i];
                Kokkos::atomic_add(&result[static_cast<size_t>(owner[i])], flux);
                Kokkos::atomic_sub(&result[static_cast<size_t>(neighbour[i])], flux);
            }
        );

        parallelFor(
            exec,
            {nInternalFaces, fnGrad.size()},
            KOKKOS_LAMBDA(const size_t i) {
                auto own = static_cast<size_t>(surfFaceCells[i - nInternalFaces]);
                scalar valueOwn = faceArea[i] * fnGrad[i];
                Kokkos::atomic_add(&result[own], valueOwn);
            }
        );

        parallelFor(
            exec,
            {0, mesh.nCells()},
            KOKKOS_LAMBDA(const size_t celli) {
                result[celli] *= operatorScaling[celli] / vol[celli];
            }
        );
    }
}

template<typename ValueType>
class GaussGreenLaplacian :
    public LaplacianOperatorFactory<ValueType>::template Register<GaussGreenLaplacian<ValueType>>
{
    using Base =
        LaplacianOperatorFactory<ValueType>::template Register<GaussGreenLaplacian<ValueType>>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Divergence"; }

    static std::string schema() { return "none"; }

    GaussGreenLaplacian(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), surfaceInterpolation_(exec, mesh, inputs),
          faceNormalGradient_(exec, mesh, inputs),
          sparsityPattern_(SparsityPattern::readOrCreate(mesh)) {};

    la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const override
    {
        la::LinearSystem<scalar, localIdx> ls(sparsityPattern_->linearSystem());

        Field<ValueType> values(ls.matrix().exec(), ls.matrix().nValues(), zero<ValueType>::value);
        Field<localIdx> mColIdxs(
            ls.matrix().exec(), ls.matrix().colIdxs().data(), ls.matrix().nColIdxs()
        );
        Field<localIdx> mRowPtrs(
            ls.matrix().exec(), ls.matrix().rowPtrs().data(), ls.matrix().rowPtrs().size()
        );

        la::CSRMatrix<ValueType, localIdx> matrix(values, mColIdxs, mRowPtrs);
        Field<ValueType> rhs(ls.matrix().exec(), ls.rhs().size(), zero<ValueType>::value);

        return {matrix, rhs, ls.sparsityPattern()};
    };

    virtual void laplacian(
        VolumeField<ValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        computeLaplacian(faceNormalGradient_, gamma, phi, lapPhi.internalField(), operatorScaling);
    };

    virtual void laplacian(
        Field<ValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        computeLaplacian(faceNormalGradient_, gamma, phi, lapPhi, operatorScaling);
    };

    virtual void laplacian(
        la::LinearSystem<ValueType, localIdx>& ls,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        const UnstructuredMesh& mesh = phi.mesh();
        const std::size_t nInternalFaces = mesh.nInternalFaces();
        const auto exec = phi.exec();
        const auto [owner, neighbour, surfFaceCells, diagOffs, ownOffs, neiOffs] = spans(
            mesh.faceOwner(),
            mesh.faceNeighbour(),
            mesh.boundaryMesh().faceCells(),
            sparsityPattern_->diagOffset(),
            sparsityPattern_->ownerOffset(),
            sparsityPattern_->neighbourOffset()
        );

        const auto [sGamma, deltaCoeffs, magFaceArea] = spans(
            gamma.internalField(),
            faceNormalGradient_.deltaCoeffs().internalField(),
            mesh.magFaceAreas()
        );

        auto [refGradient, value, valueFraction, refValue] = spans(
            phi.boundaryField().refGrad(),
            phi.boundaryField().value(),
            phi.boundaryField().valueFraction(),
            phi.boundaryField().refValue()
        );

        const auto rowPtrs = ls.matrix().rowPtrs();
        const auto colIdxs = ls.matrix().colIdxs();
        auto values = ls.matrix().values();
        auto rhs = ls.rhs().span();

        parallelFor(
            exec,
            {0, nInternalFaces},
            KOKKOS_LAMBDA(const size_t facei) {
                scalar flux = deltaCoeffs[facei] * sGamma[facei] * magFaceArea[facei];

                std::size_t own = static_cast<std::size_t>(owner[facei]);
                std::size_t nei = static_cast<std::size_t>(neighbour[facei]);

                // add neighbour contribution upper
                std::size_t rowNeiStart = rowPtrs[nei];
                std::size_t rowOwnStart = rowPtrs[own];

                // scalar valueNei = (1 - weight) * flux;
                values[rowNeiStart + neiOffs[facei]] += flux;
                Kokkos::atomic_sub(&values[rowOwnStart + diagOffs[own]], flux);

                // upper triangular part

                // add owner contribution lower
                values[rowOwnStart + ownOffs[facei]] += flux;
                Kokkos::atomic_sub(&values[rowNeiStart + diagOffs[nei]], flux);
            }
        );

        parallelFor(
            exec,
            {nInternalFaces, sGamma.size()},
            KOKKOS_LAMBDA(const size_t facei) {
                std::size_t bcfacei = facei - nInternalFaces;
                scalar flux = sGamma[facei] * magFaceArea[facei];

                std::size_t own = static_cast<std::size_t>(surfFaceCells[bcfacei]);
                std::size_t rowOwnStart = rowPtrs[own];

                values[rowOwnStart + diagOffs[own]] -=
                    flux * valueFraction[bcfacei] * deltaCoeffs[facei];
                rhs[own] -= flux
                          * (valueFraction[bcfacei] * deltaCoeffs[facei] * refValue[bcfacei]
                             + (1.0 - valueFraction[bcfacei]) * refGradient[bcfacei]);
            }
        );

        parallelFor(
            exec,
            {0, rhs.size()},
            KOKKOS_LAMBDA(const size_t celli) {
                rhs[celli] *= operatorScaling[celli];
                for (size_t i = rowPtrs[celli]; i < rowPtrs[celli + 1]; i++)
                {
                    values[i] *= operatorScaling[celli];
                }
            }
        );
    };

    std::unique_ptr<LaplacianOperatorFactory<ValueType>> clone() const override
    {
        return std::make_unique<GaussGreenLaplacian<ValueType>>(*this);
    };

private:

    SurfaceInterpolation surfaceInterpolation_;
    FaceNormalGradient faceNormalGradient_;
    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};

} // namespace NeoFOAM
