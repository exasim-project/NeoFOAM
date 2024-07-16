// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>

#include "Kokkos_Core.hpp"

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoFOAM
{

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
