// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors


#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// instantiate the template class
template class GaussGreenLaplacian<scalar>;
// template class GaussGreenLaplacian<Vector>;

};
