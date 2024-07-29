// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/stencil/geometryScheme.hpp"

#include "Kokkos_Core.hpp"

#include <functional>

namespace NeoFOAM::finiteVolume::cellCentred
{

class Upwind : public SurfaceInterpolationFactory::Register<Upwind>
{

public:

    Upwind(const Executor& exec, const UnstructuredMesh& mesh);


    static std::string name() { return "upwind"; }

    static std::string doc() { return "upwind interpolation"; }

    static std::string schema() { return "none"; }

    //    void SurfaceField<scalar> interpolate(const VolumeField<scalar>& volField ) override;

    void interpolate(const VolumeField<scalar>& volField, SurfaceField<scalar>& surfaceField) const;

    void interpolate(
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<scalar>& volField,
        SurfaceField<scalar>& surfaceField
    ) const;

    std::unique_ptr<SurfaceInterpolationFactory> clone() const override;

private:

    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

} // namespace NeoFOAM
