// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/fieldTypeDefs.hpp"
#include "NeoFOAM/fields/domainField.hpp"

#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/**
 * @brief A base class for implementing derived boundary conditions
 *
 * This class holds the information where a given boundary starts
 * and ends in the consecutive boundaryFields class
 *
 *
 * @tparam ValueType The data type of the field.
 * @note the class template parameters is currently needed since the
 * correctBoundaryConditions functions which takes templated arguments
 * is virtual.
 */
class BoundaryPatchMixin
{

public:

    BoundaryPatchMixin() {}

    BoundaryPatchMixin(const UnstructuredMesh& mesh, int patchID)
        : patchID_(patchID), start_(mesh.boundaryMesh().offset()[patchID_]),
          end_(mesh.boundaryMesh().offset()[patchID_ + 1])
    {}

    BoundaryPatchMixin(label start, label end, label patchID)
        : patchID_(patchID), start_(start), end_(end)
    {}

    label patchStart() const { return start_; };

    label patchEnd() const { return start_; };

    label patchSize() const { return end_ - start_; }

    label patchID() const { return patchID_; }

    std::pair<size_t, size_t> range() { return {start_, end_}; }

protected:

    label patchID_; ///< The id of this patch
    label start_;   ///< The start index of the patch in the boundaryField
    label end_;     ///< The end  index of the patch in the boundaryField
};
}
