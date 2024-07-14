// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
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

struct GaussGreenKernel
{
    const UnstructuredMesh& mesh_;
    const NeoFOAM::SurfaceInterpolation& surfaceInterpolation_;

    GaussGreenKernel(const UnstructuredMesh& mesh, const SurfaceInterpolation& surfInterp);

    void operator()(
        const GPUExecutor& exec,
        fvcc::VolumeField<Vector>& gradPhi,
        const fvcc::VolumeField<scalar>& phi
    );

    void operator()(
        const OMPExecutor& exec,
        fvcc::VolumeField<Vector>& gradPhi,
        const fvcc::VolumeField<scalar>& phi
    );

    void operator()(
        const CPUExecutor& exec,
        fvcc::VolumeField<Vector>& gradPhi,
        const fvcc::VolumeField<scalar>& phi
    );
};


class gaussGreenGrad
{
public:

    gaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh);

    // fvcc::VolumeField<Vector> grad(const fvcc::VolumeField<scalar>& phi);

    void grad(fvcc::VolumeField<Vector>& gradPhi, const fvcc::VolumeField<scalar>& phi);

private:

    NeoFOAM::SurfaceInterpolation surfaceInterpolation_;
    const UnstructuredMesh& mesh_;
};

} // namespace NeoFOAM
