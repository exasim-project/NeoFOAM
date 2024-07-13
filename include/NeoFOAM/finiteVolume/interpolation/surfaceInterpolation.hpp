// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

#include "Kokkos_Core.hpp"

#include <functional>


namespace NeoFOAM
{
namespace fvcc = finiteVolume::cellCentred;
class SurfaceInterpolationKernel
{

public:

    SurfaceInterpolationKernel(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual void interpolate(
        const GPUExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::VolumeField<scalar>& volField
    ) = 0;

    virtual void interpolate(
        const OMPExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::VolumeField<scalar>& volField
    ) = 0;

    virtual void interpolate(
        const CPUExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::VolumeField<scalar>& volField
    ) = 0;

    virtual void interpolate(
        const GPUExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::SurfaceField<scalar>& faceFlux,
        const fvcc::VolumeField<scalar>& volField
    ) = 0;

    virtual void interpolate(
        const OMPExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::SurfaceField<scalar>& faceFlux,
        const fvcc::VolumeField<scalar>& volField
    ) = 0;

    virtual void interpolate(
        const CPUExecutor& exec,
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::SurfaceField<scalar>& faceFlux,
        const fvcc::VolumeField<scalar>& volField
    ) = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<SurfaceInterpolationKernel> clone() const = 0;

protected:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
};

class SurfaceInterpolation
{

public:

    SurfaceInterpolation(const SurfaceInterpolation& surfInterp)
        : exec_(surfInterp.exec_), mesh_(surfInterp.mesh_),
          interpolationKernel_(surfInterp.interpolationKernel_->clone()) {};

    SurfaceInterpolation(SurfaceInterpolation&& surfInterp)
        : exec_(surfInterp.exec_), mesh_(surfInterp.mesh_),
          interpolationKernel_(std::move(surfInterp.interpolationKernel_)) {};

    SurfaceInterpolation(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::unique_ptr<SurfaceInterpolationKernel> interpolationKernel
    )
        : exec_(exec), mesh_(mesh), interpolationKernel_(std::move(interpolationKernel)) {};

    // virtual ~SurfaceInterpolationKernel() {}; // Virtual destructor

    void
    interpolate(fvcc::SurfaceField<scalar>& surfaceField, const fvcc::VolumeField<scalar>& volField) const
    {
        std::visit(
            [&](const auto& exec)
            { interpolationKernel_->interpolate(exec, surfaceField, volField); },
            exec_
        );
    }

    void interpolate(
        fvcc::SurfaceField<scalar>& surfaceField,
        const fvcc::SurfaceField<scalar>& faceFlux,
        const fvcc::VolumeField<scalar>& volField
    ) const
    {
        std::visit(
            [&](const auto& exec)
            { interpolationKernel_->interpolate(exec, surfaceField, faceFlux, volField); },
            exec_
        );
    }

private:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    std::unique_ptr<SurfaceInterpolationKernel> interpolationKernel_;
};


} // namespace NeoFOAM
