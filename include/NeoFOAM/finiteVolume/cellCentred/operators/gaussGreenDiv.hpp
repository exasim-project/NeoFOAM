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

class GaussGreenDiv : public DivOperatorFactory::Register<GaussGreenDiv>
{
public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Divergence"; }

    static std::string schema() { return "none"; }

    GaussGreenDiv(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs);

    la::LinearSystem<scalar, localIdx> createEmptyLinearSystem() const override;


    virtual void
    div(VolumeField<scalar>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<scalar>& phi,
        const dsl::Coeff operatorScaling) override;

    virtual void
    div(la::LinearSystem<scalar, localIdx>& ls,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<scalar>& phi,
        const dsl::Coeff operatorScaling) override;

    virtual void
    div(Field<scalar>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<scalar>& phi,
        const dsl::Coeff operatorScaling) override;

    virtual VolumeField<scalar>
    div(const SurfaceField<scalar>& faceFlux,
        VolumeField<scalar>& phi,
        const dsl::Coeff operatorScaling) override;

    virtual void
    div(VolumeField<Vector>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<Vector>& phi,
        const dsl::Coeff operatorScaling) override;

    virtual void
    div(Field<Vector>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<Vector>& phi,
        const dsl::Coeff operatorScaling) override;

    virtual VolumeField<Vector>
    div(const SurfaceField<scalar>& faceFlux,
        VolumeField<Vector>& phi,
        const dsl::Coeff operatorScaling) override;

    std::unique_ptr<DivOperatorFactory> clone() const override;

private:

    SurfaceInterpolation<scalar> surfaceInterpolation_;
    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};

} // namespace NeoFOAM
