// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

#include "Kokkos_Core.hpp"

#include <functional>


namespace NeoFOAM::finiteVolume::cellCentred
{

class SurfaceInterpolationFactory :
    public NeoFOAM::RuntimeSelectionFactory<
        SurfaceInterpolationFactory,
        Parameters<const Executor&, const UnstructuredMesh&>>
{

public:

    static std::string name() { return "SurfaceInterpolationFactory"; }

    SurfaceInterpolationFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~SurfaceInterpolationFactory() {} // Virtual destructor

    virtual SurfaceField<scalar> interpolate(const VolumeField<scalar>& volField) = 0;

    virtual void
    interpolate(const VolumeField<scalar>& volField, SurfaceField<scalar>& surfaceField) = 0;

    virtual void interpolate(
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<scalar>& volField,
        SurfaceField<scalar>& surfaceField
    ) = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<SurfaceInterpolationFactory> clone() const = 0;

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
        std::unique_ptr<SurfaceInterpolationFactory> interpolationKernel
    )
        : exec_(exec), mesh_(mesh), interpolationKernel_(std::move(interpolationKernel)) {};

    void interpolate(SurfaceField<scalar>& surfaceField, const VolumeField<scalar>& volField) const
    {
        interpolationKernel_->interpolate(surfaceField, volField);
    }

    void interpolate(
        SurfaceField<scalar>& surfaceField,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<scalar>& volField
    ) const
    {
        interpolationKernel_->interpolate(surfaceField, faceFlux, volField);
    }

private:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    std::unique_ptr<SurfaceInterpolationFactory> interpolationKernel_;
};


} // namespace NeoFOAM
