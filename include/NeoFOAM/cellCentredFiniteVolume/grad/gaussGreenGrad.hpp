// SPDX-License-Identifier: MIT
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

struct GaussGreenKernel
{
    const unstructuredMesh& mesh_;
    const NeoFOAM::surfaceInterpolation& surfaceInterpolation_;

    GaussGreenKernel(const unstructuredMesh& mesh, const surfaceInterpolation& surfInterp);

    void operator()(const GPUExecutor& exec, fvccVolField<Vector>& gradPhi, const fvccVolField<scalar>& phi);

    void operator()(const OMPExecutor& exec, fvccVolField<Vector>& gradPhi, const fvccVolField<scalar>& phi);

    void operator()(const CPUExecutor& exec, fvccVolField<Vector>& gradPhi, const fvccVolField<scalar>& phi);
};


class gaussGreenGrad
{
public:

    gaussGreenGrad(const executor& exec, const unstructuredMesh& mesh);

    // fvccVolField<Vector> grad(const fvccVolField<scalar>& phi);

    void grad(fvccVolField<Vector>& gradPhi, const fvccVolField<scalar>& phi);

private:

    NeoFOAM::surfaceInterpolation surfaceInterpolation_;
    const unstructuredMesh& mesh_;
};

} // namespace NeoFOAM
