// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include <functional>

#include <Kokkos_Core.hpp>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @class SurfaceInterpolationFactor
**
*/
template<typename ValueType>
class SurfaceInterpolationFactory :
    public RuntimeSelectionFactory<
        SurfaceInterpolationFactory<ValueType>,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{
    using ScalarSurfaceField = SurfaceField<scalar>;

public:

    static std::unique_ptr<SurfaceInterpolationFactory<ValueType>>
    create(const Executor& exec, const UnstructuredMesh& uMesh, const Input& inputs)
    {
        // input is dictionary the key is "interpolation"
        std::string key =
            (std::holds_alternative<NeoN::Dictionary>(inputs))
                ? std::get<NeoN::Dictionary>(inputs).get<std::string>("surfaceInterpolation")
                : std::get<NeoN::TokenList>(inputs).next<std::string>();

        SurfaceInterpolationFactory<ValueType>::keyExistsOrError(key);
        return SurfaceInterpolationFactory<ValueType>::table().at(key)(exec, uMesh, inputs);
    }

    static std::string name() { return "SurfaceInterpolationFactory"; }

    SurfaceInterpolationFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~SurfaceInterpolationFactory() {} // Virtual destructor

    virtual void
    interpolate(const VolumeField<ValueType>& src, SurfaceField<ValueType>& dst) const = 0;

    virtual void interpolate(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& dst
    ) const = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<SurfaceInterpolationFactory<ValueType>> clone() const = 0;

protected:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
};

template<typename ValueType>
class SurfaceInterpolation
{

    using FieldValueType = ValueType;

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
        std::unique_ptr<SurfaceInterpolationFactory<ValueType>> interpolationKernel
    )
        : exec_(exec), mesh_(mesh), interpolationKernel_(std::move(interpolationKernel)) {};

    SurfaceInterpolation(const Executor& exec, const UnstructuredMesh& mesh, const Input& input)
        : exec_(exec), mesh_(mesh),
          interpolationKernel_(SurfaceInterpolationFactory<ValueType>::create(exec, mesh, input)) {
          };


    void interpolate(const VolumeField<ValueType>& src, SurfaceField<ValueType>& dst) const
    {
        interpolationKernel_->interpolate(src, dst);
    }

    void interpolate(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& dst
    ) const
    {
        interpolationKernel_->interpolate(flux, src, dst);
    }

    SurfaceField<ValueType> interpolate(const VolumeField<ValueType>& src) const
    {
        std::string nameInterpolated = "interpolated_" + src.name;
        SurfaceField<ValueType> dst(
            exec_, nameInterpolated, mesh_, createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh_)
        );
        interpolate(dst, src);
        return dst;
    }

    SurfaceField<ValueType>
    interpolate(const SurfaceField<ValueType>& flux, const VolumeField<ValueType>& src) const
    {
        std::string name = "interpolated_" + src.name;
        SurfaceField<ValueType> dst(
            exec_, name, mesh_, createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh_)
        );
        interpolate(flux, src, dst);
        return dst;
    }

private:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> interpolationKernel_;
};


} // namespace NeoN
