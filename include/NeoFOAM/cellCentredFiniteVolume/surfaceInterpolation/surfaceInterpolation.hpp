// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/fields/fvccVolField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/fields/fvccSurfaceField.hpp"
// #include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolationSelector.hpp"
#include "Kokkos_Core.hpp"
#include <functional>


namespace NeoFOAM
{

class surfaceInterpolationKernel
{

public:

    surfaceInterpolationKernel(const executor& exec, const unstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual void interpolate(const GPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField) = 0;
    virtual void interpolate(const OMPExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField) = 0;
    virtual void interpolate(const CPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField) = 0;

    virtual void interpolate(const GPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& volField) = 0;
    virtual void interpolate(const OMPExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& volField) = 0;
    virtual void interpolate(const CPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& volField) = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<surfaceInterpolationKernel> clone() const = 0;

protected:

    const executor exec_;
    const unstructuredMesh& mesh_;
};

class surfaceInterpolation
{
public:

    surfaceInterpolation(const surfaceInterpolation& surfInterp)
        : exec_(surfInterp.exec_), mesh_(surfInterp.mesh_), interpolationKernel_(surfInterp.interpolationKernel_->clone()) {};

    surfaceInterpolation(surfaceInterpolation&& surfInterp)
        : exec_(surfInterp.exec_), mesh_(surfInterp.mesh_), interpolationKernel_(std::move(surfInterp.interpolationKernel_)) {};

    surfaceInterpolation(const executor& exec, const unstructuredMesh& mesh, std::unique_ptr<surfaceInterpolationKernel> interpolationKernel)
        : exec_(exec), mesh_(mesh), interpolationKernel_(std::move(interpolationKernel)) {};

    // virtual ~surfaceInterpolationKernel() {}; // Virtual destructor

    void interpolate(fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField) const
    {
        std::visit([&](const auto& exec)
                   { interpolationKernel_->interpolate(exec, surfaceField, volField); },
                   exec_);
    }

    void interpolate(fvccSurfaceField<scalar>& surfaceField, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& volField) const
    {
        std::visit([&](const auto& exec)
                   { interpolationKernel_->interpolate(exec, surfaceField, faceFlux, volField); },
                   exec_);
    }

private:

    const executor exec_;
    const unstructuredMesh& mesh_;
    std::unique_ptr<surfaceInterpolationKernel> interpolationKernel_;
};


} // namespace NeoFOAM
