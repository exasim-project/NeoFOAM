// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>

#include "Kokkos_Core.hpp"

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

class GaussGreenGrad
{
public:

    GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh);

    // fvcc::VolumeField<Vector> grad(const fvcc::VolumeField<scalar>& phi);

    void grad(VolumeField<Vector>& gradPhi, const VolumeField<scalar>& phi);

private:

    SurfaceInterpolation surfaceInterpolation_;
    const UnstructuredMesh& mesh_;
};

} // namespace NeoFOAM
