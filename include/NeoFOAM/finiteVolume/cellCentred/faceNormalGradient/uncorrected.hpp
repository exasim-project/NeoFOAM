// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


namespace NeoFOAM::finiteVolume::cellCentred
{


class Uncorrected : public FaceNormalGradientFactory::Register<Uncorrected>
{

public:

    Uncorrected(const Executor& exec, const UnstructuredMesh& mesh, Input input);

    Uncorrected(const Executor& exec, const UnstructuredMesh& mesh);

    static std::string name() { return "uncorrected"; }

    static std::string doc() { return "Uncorrected interpolation"; }

    static std::string schema() { return "none"; }

    virtual void faceNormalGrad(
        const VolumeField<scalar>& volField, SurfaceField<scalar>& surfaceField
    ) const override;

    virtual const SurfaceField<scalar>& deltaCoeffs() const override;

    std::unique_ptr<FaceNormalGradientFactory> clone() const override;

private:

    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

} // namespace NeoFOAM
