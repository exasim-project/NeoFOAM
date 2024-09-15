// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/inputs.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


namespace NeoFOAM::finiteVolume::cellCentred
{

class SurfaceInterpolationFactory :
    public NeoFOAM::RuntimeSelectionFactory<
        SurfaceInterpolationFactory,
        Parameters<const Executor&, const UnstructuredMesh&, Input>>
{

public:

    static std::unique_ptr<SurfaceInterpolationFactory>
    create(const Executor& exec, const UnstructuredMesh& uMesh, Input inputs)
    {
        std::string key;
        // input is dictionary the key is "interpolation"
        if (std::holds_alternative<NeoFOAM::Dictionary>(inputs))
        {
            key = std::get<NeoFOAM::Dictionary>(inputs).get<std::string>("surfaceInterpolation");
        }
        else
        {
            key = std::get<NeoFOAM::TokenList>(inputs).get<std::string>(0);
        }

        keyExistsOrError(key);
        auto ptr = table().at(key)(exec, uMesh, inputs);
        return ptr;
    }

    static std::string name() { return "SurfaceInterpolationFactory"; }

    SurfaceInterpolationFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~SurfaceInterpolationFactory() {} // Virtual destructor

    virtual void
    interpolate(SurfaceField<scalar>& surfaceField, const VolumeField<scalar>& volField) = 0;


    virtual void interpolate(
        SurfaceField<scalar>& surfaceField,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<scalar>& volField
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

    SurfaceInterpolation(const Executor& exec, const UnstructuredMesh& mesh, Input input)
        : exec_(exec), mesh_(mesh),
          interpolationKernel_(SurfaceInterpolationFactory::create(exec, mesh, input)) {};

    void interpolate(SurfaceField<scalar>& surfaceField, const VolumeField<scalar>& volField) const
    {
        interpolationKernel_->interpolate(surfaceField, volField);
    }

    SurfaceField<scalar> interpolate(const VolumeField<scalar>& volField) const
    {
        SurfaceField<scalar> surfaceField(
            exec_, "phif", mesh_, SurfaceBoundary<scalar>::calculatedBCs(mesh_)
        );
        interpolate(surfaceField, volField);
        return surfaceField;
    }

    void interpolate(
        SurfaceField<scalar>& surfaceField,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<scalar>& volField
    ) const
    {
        interpolationKernel_->interpolate(surfaceField, faceFlux, volField);
    }

    SurfaceField<scalar>
    interpolate(const SurfaceField<scalar>& faceFlux, const VolumeField<scalar>& volField) const
    {
        SurfaceField<scalar> surfaceField(
            exec_, "phif", mesh_, SurfaceBoundary<scalar>::calculatedBCs(mesh_)
        );
        interpolate(surfaceField, faceFlux, volField);
        return surfaceField;
    }

private:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    std::unique_ptr<SurfaceInterpolationFactory> interpolationKernel_;
};


} // namespace NeoFOAM