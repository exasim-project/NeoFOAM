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

/**
 * @brief Represents a surface field for a cell-centered finite volume method.
 *
 * This class stores the surface field data for a cell-centered finite volume method. It contains
 * the internal field and boundary field, as well as the boundary conditions.
 *
 * @tparam ValueType The data type of the field.
 */
template<typename ValueType>
class SurfaceField final : public GeometricField<ValueType, SurfaceBoundaryField<ValueType>>
{
public:

    using GeometricField<ValueType, SurfaceBoundaryField<ValueType>>::GeometricField;
};

} // namespace NeoFOAM
