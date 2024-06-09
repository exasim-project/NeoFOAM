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
 * @class VolumeField
 * @brief Represents a cell-centered finite volume field at cell centers.
 *
 * The VolumeField class is used to store field information at the cell center.
 * It provides methods to correct boundary conditions, access the internal field, boundary field,
 * and executor, and retrieve information about the mesh and boundary conditions.
 *
 * @tparam ValueType The type of the field values.
 */
template<typename ValueType>
class VolumeField final : public GeometricField<ValueType, BoundaryField<ValueType>>
{
public:

    using GeometricField<ValueType, BoundaryField<ValueType>>::GeometricField;
};

} // namespace NeoFOAM
