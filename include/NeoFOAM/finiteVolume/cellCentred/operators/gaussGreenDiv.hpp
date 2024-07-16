// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include "Kokkos_Core.hpp"

#include <functional>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoFOAM
{

class GaussGreenDiv
{
public:

    GaussGreenDiv(
        const Executor& exec, const UnstructuredMesh& mesh, const SurfaceInterpolation& surfInterp
    );

    void
    div(fvcc::VolumeField<scalar>& divPhi,
        const fvcc::SurfaceField<scalar>& faceFlux,
        fvcc::VolumeField<scalar>& phi);

private:

    const UnstructuredMesh& mesh_;
    SurfaceInterpolation surfaceInterpolation_;
};

} // namespace NeoFOAM
