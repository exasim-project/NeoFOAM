// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>

#include <Kokkos_Core.hpp>

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/core/runtimeSelectionFactory.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/* @class SurfaceInterpolationFactor
**
*/
class SurfaceInterpolationFactory :
    public RuntimeSelectionFactory<
        SurfaceInterpolationFactory,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{
    using ScalarSurfaceField = SurfaceField<scalar>;

public:

    static std::unique_ptr<SurfaceInterpolationFactory>
    create(const Executor& exec, const UnstructuredMesh& uMesh, const Input& inputs)
    {
        // input is dictionary the key is "interpolation"
        std::string key =
            (std::holds_alternative<NeoFOAM::Dictionary>(inputs))
                ? std::get<NeoFOAM::Dictionary>(inputs).get<std::string>("surfaceInterpolation")
                : std::get<NeoFOAM::TokenList>(inputs).next<std::string>();

        keyExistsOrError(key);
        return table().at(key)(exec, uMesh, inputs);
    }

    static std::string name() { return "SurfaceInterpolationFactory"; }

    SurfaceInterpolationFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~SurfaceInterpolationFactory() {} // Virtual destructor

    virtual void interpolate(const VolumeField<scalar>& src, SurfaceField<scalar>& dst) const = 0;

    virtual void interpolate(
        const ScalarSurfaceField& flux, const VolumeField<scalar>& src, SurfaceField<scalar>& dst
    ) const = 0;

    virtual void interpolate(const VolumeField<Vector>& src, SurfaceField<Vector>& dst) const = 0;

    virtual void interpolate(
        const ScalarSurfaceField& flux, const VolumeField<Vector>& src, SurfaceField<Vector>& dst
    ) const = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<SurfaceInterpolationFactory> clone() const = 0;

protected:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
};

class SurfaceInterpolation
{
    using ScalarSurfaceField = SurfaceField<scalar>;

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

    SurfaceInterpolation(const Executor& exec, const UnstructuredMesh& mesh, const Input& input)
        : exec_(exec), mesh_(mesh),
          interpolationKernel_(SurfaceInterpolationFactory::create(exec, mesh, input)) {};


    void interpolate(const VolumeField<scalar>& src, ScalarSurfaceField& dst) const
    {
        interpolationKernel_->interpolate(src, dst);
    }

    void interpolate(const VolumeField<Vector>& src, SurfaceField<Vector>& dst) const
    {
        interpolationKernel_->interpolate(src, dst);
    }


    void interpolate(
        const ScalarSurfaceField& flux, const VolumeField<scalar>& src, ScalarSurfaceField& dst
    ) const
    {
        interpolationKernel_->interpolate(flux, src, dst);
    }

    void interpolate(
        const ScalarSurfaceField& flux, const VolumeField<Vector>& src, SurfaceField<Vector>& dst
    ) const
    {
        interpolationKernel_->interpolate(flux, src, dst);
    }

    ScalarSurfaceField interpolate(const VolumeField<scalar>& src) const
    {
        std::string nameInterpolated = "interpolated_" + src.name;
        ScalarSurfaceField dst(
            exec_, nameInterpolated, mesh_, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh_)
        );
        interpolate(dst, src);
        return dst;
    }

    ScalarSurfaceField
    interpolate(const ScalarSurfaceField& flux, const VolumeField<scalar>& src) const
    {
        std::string name = "interpolated_" + src.name;
        ScalarSurfaceField dst(
            exec_, name, mesh_, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh_)
        );
        interpolate(flux, src, dst);
        return dst;
    }

private:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    std::unique_ptr<SurfaceInterpolationFactory> interpolationKernel_;
};


} // namespace NeoFOAM
