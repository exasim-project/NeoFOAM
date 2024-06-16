// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <vector>

#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/boundaryStrategy.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
using VolumeField = GeometricField<ValueType, BoundaryFactory<ValueType, int>>;

} // namespace NeoFOAM
