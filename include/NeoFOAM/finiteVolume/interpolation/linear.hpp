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

class linear : public surfaceInterpolationKernel
{

public:

    linear(const executor& exec, const unstructuredMesh& mesh);

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

    std::unique_ptr<surfaceInterpolationKernel> clone() const override;

    static std::unique_ptr<surfaceInterpolationKernel>
    Create(const executor& exec, const unstructuredMesh& mesh);

private:

    const unstructuredMesh& mesh_;
    // const FvccGeometryScheme geometryScheme_;
    const std::shared_ptr<FvccGeometryScheme> geometryScheme_;

    static bool s_registered;
};


} // namespace NeoFOAM
