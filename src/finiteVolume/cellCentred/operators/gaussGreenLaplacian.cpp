// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


GaussGreenLaplacian::GaussGreenLaplacian(
    const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs
)
    : LaplacianOperatorFactory::Register<GaussGreenLaplacian>(exec, mesh),
      surfaceInterpolation_(exec, mesh, inputs),
      sparsityPattern_(SparsityPattern::readOrCreate(mesh)) {};


la::LinearSystem<scalar, localIdx> GaussGreenLaplacian::createEmptyLinearSystem() const
{
    return sparsityPattern_->linearSystem();
};

void GaussGreenLaplacian::laplacian(
    VolumeField<scalar>& lapPhi, const SurfaceField<scalar>& gamma, VolumeField<scalar>& phi
) {
    // computeDiv(faceFlux, phi, surfaceInterpolation_, divPhi);
};


std::unique_ptr<LaplacianOperatorFactory> GaussGreenLaplacian::clone() const
{
    return std::make_unique<GaussGreenLaplacian>(*this);
}


};
