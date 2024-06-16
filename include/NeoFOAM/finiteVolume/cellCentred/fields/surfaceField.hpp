// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <vector>

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/fields/domainField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
using SurfaceField = GeometricField<ValueType, BoundaryFactory<ValueType, int>>;

} // namespace NeoFOAM
