// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"
#include "NeoFOAM/mesh/stencil/FvccGeometryScheme.hpp"

#include "Kokkos_Core.hpp"

#include <functional>

namespace NeoFOAM
{

class Upwind : public SurfaceInterpolationKernel
{

public:

    Upwind(const executor& exec, const UnstructuredMesh& mesh);

    void interpolate(
        const GPUExecutor& exec,
        fvccSurfaceField<scalar>& surfaceField,
        const fvccVolField<scalar>& volField
    );

    void interpolate(
        const OMPExecutor& exec,
        fvccSurfaceField<scalar>& surfaceField,
        const fvccVolField<scalar>& volField
    );

    void interpolate(
        const CPUExecutor& exec,
        fvccSurfaceField<scalar>& surfaceField,
        const fvccVolField<scalar>& volField
    );

    void interpolate(
        const GPUExecutor& exec,
        fvccSurfaceField<scalar>& surfaceField,
        const fvccSurfaceField<scalar>& faceFlux,
        const fvccVolField<scalar>& volField
    );

    void interpolate(
        const OMPExecutor& exec,
        fvccSurfaceField<scalar>& surfaceField,
        const fvccSurfaceField<scalar>& faceFlux,
        const fvccVolField<scalar>& volField
    );

    void interpolate(
        const CPUExecutor& exec,
        fvccSurfaceField<scalar>& surfaceField,
        const fvccSurfaceField<scalar>& faceFlux,
        const fvccVolField<scalar>& volField
    );

    std::unique_ptr<SurfaceInterpolationKernel> clone() const override;

    static std::unique_ptr<SurfaceInterpolationKernel>
    create(const executor& exec, const UnstructuredMesh& mesh);

private:

    static bool sSegistered;

    const UnstructuredMesh& mesh_;
    // const FvccGeometryScheme geometryScheme_;
    const std::shared_ptr<FvccGeometryScheme> geometryScheme_;
};


} // namespace NeoFOAM
