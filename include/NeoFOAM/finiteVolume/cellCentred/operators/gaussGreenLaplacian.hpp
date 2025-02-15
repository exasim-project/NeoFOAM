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

class GaussGreenLaplacian : public LaplacianOperatorFactory::Register<GaussGreenLaplacian>
{
public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Divergence"; }

    static std::string schema() { return "none"; }

    GaussGreenLaplacian(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs);

    la::LinearSystem<scalar, localIdx> createEmptyLinearSystem() const override;

    virtual void laplacian(
        VolumeField<scalar>& lapPhi, const SurfaceField<scalar>& gamma, VolumeField<scalar>& phi
    ) override;

    virtual void laplacian(
        Field<scalar>& lapPhi, const SurfaceField<scalar>& gamma, VolumeField<scalar>& phi
    ) override;

    std::unique_ptr<LaplacianOperatorFactory> clone() const override;

private:

    SurfaceInterpolation surfaceInterpolation_;
    FaceNormalGradient faceNormalGradient_;
    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};

} // namespace NeoFOAM
