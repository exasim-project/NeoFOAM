// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/fields/fvccVolField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/fields/fvccSurfaceField.hpp"
#include "Kokkos_Core.hpp"
#include <functional>


namespace NeoFOAM
{

class surfaceInterpolationKernel
{

public:

    surfaceInterpolationKernel(const executor& exec, const unstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual void operator()(const GPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField) = 0;
    virtual void operator()(const OMPExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField) = 0;
    virtual void operator()(const CPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField) = 0;

protected:

    const executor exec_;
    const unstructuredMesh& mesh_;
};

class surfaceInterpolation
{
public:

    surfaceInterpolation(const executor& exec, const unstructuredMesh& mesh, std::unique_ptr<surfaceInterpolationKernel> interpolationKernel)
        : exec_(exec), mesh_(mesh), interpolationKernel_(std::move(interpolationKernel)) {};

    // virtual ~surfaceInterpolationKernel() {}; // Virtual destructor

    void interpolate(fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField)
    {
        std::visit([&](const auto& exec)
                   { interpolationKernel_->operator()(exec, surfaceField, volField); },
                   exec_);
    }

private:

    const executor exec_;
    const unstructuredMesh& mesh_;
    std::unique_ptr<surfaceInterpolationKernel> interpolationKernel_;
};

} // namespace NeoFOAM