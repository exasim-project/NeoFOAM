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
 * @brief Represents a boundary field for a cell-centered finite volume method.
 *
 * @tparam ValueType The data type of the field.
 */
template<typename ValueType>
class BoundaryField
{

public:

    BoundaryField(Executor exec) : exec_(exec), mesh_(nullptr) {}

    BoundaryField(Executor exec, std::shared_ptr<const UnstructuredMesh> mesh, int patchID)
        : exec_(exec), mesh_(mesh), patchID_(patchID),
          start_(mesh->boundaryMesh().offset()[patchID_]),
          end_(mesh->boundaryMesh().offset()[patchID_ + 1])
    {}

    virtual void correctBoundaryConditions(
        BoundaryFields<ValueType>& bfield, const Field<ValueType>& internalField
    )
    {}

    label start() const { return start_; };

    label end() const { return end_; };

    label size() const { return end_ - start_; }

    label patchID() const { return patchID_; }

    const Executor& exec() const { return exec_; }

protected:

    Executor exec_; // The executor object
    std::shared_ptr<const UnstructuredMesh> mesh_;
    label patchID_;
    label start_;
    label end_;
};
};
