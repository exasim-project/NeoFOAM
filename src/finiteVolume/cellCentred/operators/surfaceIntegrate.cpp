// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors


#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// instantiate the template class
template class SurfaceIntegrate<scalar>;
template class SurfaceIntegrate<Vector>;

};
