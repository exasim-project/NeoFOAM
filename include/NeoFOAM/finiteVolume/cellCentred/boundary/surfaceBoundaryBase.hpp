// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/domainField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/boundaryPatchMixin.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/**
 * @brief Represents a surface boundary field for a cell-centered finite volume method.
 *
 * @tparam ValueType The data type of the field.
 */
template<typename ValueType>
class SurfaceBoundaryPatchMixin : public BoundaryPatchMixin
{

    virtual void correctBoundaryCondition(DomainField<ValueType>& domainField) {}
};

}
