// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/interpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/mesh/stencil/fvccGeometryScheme.hpp"

#include "Kokkos_Core.hpp"

#include <functional>


namespace NeoFOAM
{

class Linear : public SurfaceInterpolationKernel
{

public:

    Linear(const Executor& exec, const UnstructuredMesh& mesh);

    void interpolate(
        const GPUExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::VolumeField<scalar>& volField
    );

    void interpolate(
        const OMPExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::VolumeField<scalar>& volField
    );

    void interpolate(
        const CPUExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::VolumeField<scalar>& volField
    );

    void interpolate(
        const GPUExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::SurfaceField<scalar>& faceFlux,
        const fvcc::VolumeField<scalar>& volField
    );

    void interpolate(
        const OMPExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::SurfaceField<scalar>& faceFlux,
        const fvcc::VolumeField<scalar>& volField
    );

    void interpolate(
        const CPUExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::SurfaceField<scalar>& faceFlux,
        const fvcc::VolumeField<scalar>& volField
    );

    std::unique_ptr<SurfaceInterpolationKernel> clone() const override;

    static std::unique_ptr<SurfaceInterpolationKernel>
    create(const Executor& exec, const UnstructuredMesh& mesh);

private:

    const UnstructuredMesh& mesh_;
    // const FvccGeometryScheme geometryScheme_;
    const std::shared_ptr<FvccGeometryScheme> geometryScheme_;

    static bool sRegistered;
};


} // namespace NeoFOAM