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


std::unique_ptr<LaplacianOperatorFactory> GaussGreenLaplacian::clone() const
{
    return std::make_unique<GaussGreenLaplacian>(*this);
}


};
