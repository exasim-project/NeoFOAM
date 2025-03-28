// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors


#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/expression.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// instantiate the template class
template class Expression<scalar>;
// template class Expression<Vector>;

};
