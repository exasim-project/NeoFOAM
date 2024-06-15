// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/fieldTypeDefs.hpp"
#include "NeoFOAM/fields/boundaryFields.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/**
 * @brief Represents a surface boundary field for a cell-centered finite volume method.
 *
 * @tparam ValueType The data type of the field.
 */
template<typename ValueType>
class SurfaceBoundaryBase : public BoundaryBase<ValueType>
{
    using BoundaryBase<ValueType>::BoundaryBase;

    virtual void correctBoundaryConditions(DomainField<ValueType>& domainField) {}
};

}
