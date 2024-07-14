// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

#include "Kokkos_Core.hpp"
#include <functional>

#include "NeoFOAM/finiteVolume/interpolation/linear.hpp"
#include "NeoFOAM/finiteVolume/interpolation/upwind.hpp"
#include "NeoFOAM/finiteVolume/interpolation/surfaceInterpolation.hpp"

namespace NeoFOAM
{

struct GaussGreenDivKernel
{
    const UnstructuredMesh& mesh_;

    const NeoFOAM::SurfaceInterpolation& surfaceInterpolation_;

    GaussGreenDivKernel(const UnstructuredMesh& mesh, const SurfaceInterpolation& surfInterp);

    void operator()(
        const GPUExecutor& exec,
        fvcc::VolumeField<scalar>& divPhi,
        const fvcc::SurfaceField<scalar>& faceFlux,
        const fvcc::VolumeField<scalar>& phi
    );

    void operator()(
        const OMPExecutor& exec,
        fvcc::VolumeField<scalar>& divPhi,
        const fvcc::SurfaceField<scalar>& faceFlux,
        const fvcc::VolumeField<scalar>& phi
    );

    void operator()(
        const CPUExecutor& exec,
        fvcc::VolumeField<scalar>& divPhi,
        const fvcc::SurfaceField<scalar>& faceFlux,
        const fvcc::VolumeField<scalar>& phi
    );
};


class GaussGreenDiv
{
public:

    GaussGreenDiv(
        const Executor& exec, const UnstructuredMesh& mesh, const SurfaceInterpolation& surfInterp
    );

    // fvcc::VolumeField<scalar> grad(const fvcc::VolumeField<scalar>& phi);

    void
    div(fvcc::VolumeField<scalar>& divPhi,
        const fvcc::SurfaceField<scalar>& faceFlux,
        fvcc::VolumeField<scalar>& phi);

private:

    const UnstructuredMesh& mesh_;

    SurfaceInterpolation surfaceInterpolation_;
};

} // namespace NeoFOAM
