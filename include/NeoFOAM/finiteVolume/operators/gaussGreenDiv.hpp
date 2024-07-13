// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

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
    const UnstructuredMesh& mesh_;

    const NeoFOAM::surfaceInterpolation& surfaceInterpolation_;

    GaussGreenDivKernel(const UnstructuredMesh& mesh, const SurfaceInterpolation& surfInterp);

    void operator()(
        const Executor& exec,
        fvccVolField<scalar>& divPhi,
        const fvccSurfaceField<scalar>& faceFlux,
        const fvccVolField<scalar>& phi
    );
};


class GaussGreenDiv
{
public:

    gaussGreenDiv(
        const Executor& exec, const UnstructuredMesh& mesh, const SurfaceInterpolation& surfInterp
    );

    // fvccVolField<scalar> grad(const fvccVolField<scalar>& phi);

    void
    div(fvccVolField<scalar>& divPhi,
        const fvccSurfaceField<scalar>& faceFlux,
        fvccVolField<scalar>& phi);

private:

    const UnstructuredMesh& mesh_;

    SurfaceInterpolation surfaceInterpolation_;
};

} // namespace NeoFOAM
