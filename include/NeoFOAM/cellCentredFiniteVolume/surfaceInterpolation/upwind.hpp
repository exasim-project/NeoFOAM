// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"
#include "Kokkos_Core.hpp"
#include <functional>


namespace NeoFOAM
{

class upwind :
    public surfaceInterpolationKernel
{

public:

    upwind(const executor& exec, const unstructuredMesh& mesh)
        : surfaceInterpolationKernel(exec,mesh) {};

    void operator()(const GPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField)
    {
        std::cout << "upwind GPU" << std::endl;
    }

    void operator()(const OMPExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField)
    {
        std::cout << "upwind OMP" << std::endl;
    }

    void operator()(const CPUExecutor& exec, fvccSurfaceField<scalar>& surfaceField, const fvccVolField<scalar>& volField)
    {
        std::cout << "upwind CPU" << std::endl;
    }

private:
};


} // namespace NeoFOAM