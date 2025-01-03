// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

class GaussGreenDiv : public DivOperatorFactory::Register<GaussGreenDiv>
{
public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Divergence"; }

    static std::string schema() { return "none"; }

    GaussGreenDiv(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs);

    void
    div(VolumeField<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
    ) override;

    void
    div(Field<scalar>& divPhi, const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi
    ) override;

    VolumeField<scalar>
    div(const SurfaceField<scalar>& faceFlux, VolumeField<scalar>& phi) override;

    std::unique_ptr<DivOperatorFactory> clone() const override;

private:

    SurfaceInterpolation surfaceInterpolation_;
};

} // namespace NeoFOAM
