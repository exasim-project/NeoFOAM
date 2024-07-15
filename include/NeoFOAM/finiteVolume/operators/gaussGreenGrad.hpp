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
namespace detail
{
void computeGrad(
    fvcc::VolumeField<Vector>& gradPhi,
    const fvcc::VolumeField<scalar>& phi,
    const SurfaceInterpolation& surfInterp
);

}


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
