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

#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/linear.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/upwind.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolation.hpp"

namespace NeoFOAM
{

struct GaussGreenDivKernel
{
    const unstructuredMesh& mesh_;
    const NeoFOAM::surfaceInterpolation& surfaceInterpolation_;

    GaussGreenDivKernel(const unstructuredMesh& mesh, const surfaceInterpolation& surfInterp);

    void operator()(const GPUExecutor& exec, fvccVolField<scalar>& divPhi, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& phi);

    void operator()(const OMPExecutor& exec, fvccVolField<scalar>& divPhi, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& phi);

    void operator()(const CPUExecutor& exec, fvccVolField<scalar>& divPhi, const fvccSurfaceField<scalar>& faceFlux, const fvccVolField<scalar>& phi);
};


class gaussGreenDiv
{
public:

    gaussGreenDiv(const executor& exec, const unstructuredMesh& mesh);

    // fvccVolField<scalar> grad(const fvccVolField<scalar>& phi);

    void div(fvccVolField<scalar>& divPhi, const fvccSurfaceField<scalar>& faceFlux, fvccVolField<scalar>& phi);

private:

    const unstructuredMesh& mesh_;
    surfaceInterpolation surfaceInterpolation_;
};

} // namespace NeoFOAM