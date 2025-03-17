// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <memory>

#include "NeoFOAM/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

// instantiate the template class
template class Uncorrected<scalar>;
template class Uncorrected<Vector>;

} // namespace NeoFOAM
