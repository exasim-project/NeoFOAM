// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/fields/BoundaryFields.hpp"
#include "NeoFOAM/mesh/unstructured/UnstructuredMesh.hpp"

namespace NeoFOAM
{

template<typename T>
class fvccSurfaceBoundaryField
{

public:

    fvccSurfaceBoundaryField(const UnstructuredMesh& mesh, int patchID)
        : mesh_(mesh), patchID_(patchID), start_(mesh.boundaryMesh().offset()[patchID_]),
          end_(mesh.boundaryMesh().offset()[patchID_ + 1]), size_(end_ - start_) {

                                                            };

    virtual void
    correctBoundaryConditions(BoundaryFields<scalar>& bfield, Field<scalar>& internalField) {

    };

    int size() const { return size_; }

protected:

    const UnstructuredMesh& mesh_;
    int patchID_;
    int start_;
    int end_;
    int size_;
};

};
