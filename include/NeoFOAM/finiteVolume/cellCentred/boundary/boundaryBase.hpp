// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/fields/fieldTypeDefs.hpp"
#include "NeoFOAM/fields/boundaryFields.hpp"
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
 * NOTE the class template parameters is currently needed since the
 * correctBoundaryConditions functions which takes templated arguments
 * is virtual.
 */
template<typename ValueType>
class BoundaryBase
{

public:

    BoundaryBase(Executor exec) : exec_(exec), mesh_(nullptr) {}

    BoundaryBase(Executor exec, std::shared_ptr<const UnstructuredMesh> mesh, int patchID)
        : exec_(exec), mesh_(mesh), patchID_(patchID),
          start_(mesh->boundaryMesh().offset()[patchID_]),
          end_(mesh->boundaryMesh().offset()[patchID_ + 1])
    {}

    /**
     * @brief Update corresponding boundary state
     */
    virtual void correctBoundaryConditions(
        const Field<ValueType>& internalField, BoundaryFields<ValueType>& bfield
    )
    {}

    label start() const { return start_; };

    label end() const { return start_; };

    label size() const { return end_ - start_; }

    label patchID() const { return patchID_; }

    const Executor& exec() const { return exec_; }

protected:

    Executor exec_; ///< The executor object
    std::shared_ptr<const UnstructuredMesh> mesh_;
    label patchID_; ///< The id of this patch
    label start_;   ///< The start index of the patch in the boundaryField
    label end_;     ///< The end  index of the patch in the boundaryField
};
}
