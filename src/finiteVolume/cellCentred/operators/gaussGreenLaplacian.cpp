// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

void computeLaplacian(
    const FaceNormalGradient& faceNormalGradient,
    const SurfaceField<scalar>& gamma,
    VolumeField<scalar>& phi,
    Field<scalar>& lapPhi
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
            result[celli] *= 1 / vol[celli];
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
            KOKKOS_LAMBDA(const size_t celli) { result[celli] *= 1 / vol[celli]; }
        );
    }
}


GaussGreenLaplacian::GaussGreenLaplacian(
    const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs
)
    : LaplacianOperatorFactory::Register<GaussGreenLaplacian>(exec, mesh),
      surfaceInterpolation_(exec, mesh, inputs), faceNormalGradient_(exec, mesh, inputs),
      sparsityPattern_(SparsityPattern::readOrCreate(mesh)) {

      };


la::LinearSystem<scalar, localIdx> GaussGreenLaplacian::createEmptyLinearSystem() const
{
    return sparsityPattern_->linearSystem();
};

void GaussGreenLaplacian::laplacian(
    VolumeField<scalar>& lapPhi, const SurfaceField<scalar>& gamma, VolumeField<scalar>& phi
)
{
    computeLaplacian(faceNormalGradient_, gamma, phi, lapPhi.internalField());
};

void GaussGreenLaplacian::laplacian(
    Field<scalar>& lapPhi, const SurfaceField<scalar>& gamma, VolumeField<scalar>& phi
)
{
    computeLaplacian(faceNormalGradient_, gamma, phi, lapPhi);
};

void GaussGreenLaplacian::laplacian(
    la::LinearSystem<scalar, localIdx>& ls,
    const SurfaceField<scalar>& gamma,
    VolumeField<scalar>& phi
)
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
    // valueFraction_* this->patch().deltaCoeffs() * refValue_ + (1.0 - valueFraction_) * refGrad_;
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
};


std::unique_ptr<LaplacianOperatorFactory> GaussGreenLaplacian::clone() const
{
    return std::make_unique<GaussGreenLaplacian>(*this);
}


};
