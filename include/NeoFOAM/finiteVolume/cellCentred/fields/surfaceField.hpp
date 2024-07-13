// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <vector>

#include "NeoFOAM/finiteVolume/cellCentred/fields/geometricField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
using SurfaceField = GeometricField<ValueType, SurfaceBoundary<ValueType>>;

} // namespace NeoFOAM
