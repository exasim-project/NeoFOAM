// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


namespace NeoFOAM::finiteVolume::cellCentred
{


class Linear : public SurfaceInterpolationFactory::Register<Linear>
{

public:

    Linear(const Executor& exec, const UnstructuredMesh& mesh, Input input);

    Linear(const Executor& exec, const UnstructuredMesh& mesh);

    static std::string name() { return "linear"; }

    static std::string doc() { return "linear interpolation"; }

    static std::string schema() { return "none"; }

    void interpolate(const VolumeField<scalar>& volField, SurfaceField<scalar>& surfaceField)
        const override;

    void interpolate(
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<scalar>& volField,
        SurfaceField<scalar>& surfaceField
    ) const override;

    std::unique_ptr<SurfaceInterpolationFactory> clone() const override;

private:

    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

} // namespace NeoFOAM
