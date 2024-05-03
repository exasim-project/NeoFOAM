// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"
#include "NeoFOAM/fields/boundaryFields.hpp"

namespace NeoFOAM
{

template<typename T>
class fvccSurfaceBoundaryField
{

public:

    fvccSurfaceBoundaryField(const unstructuredMesh& mesh, int patchID)
        : mesh_(mesh),
          patchID_(patchID),
          start_(mesh.boundaryMesh().offset()[patchID_]),
          end_(mesh.boundaryMesh().offset()[patchID_ + 1]),
          size_(end_ - start_) {

          };

    virtual void correctBoundaryConditions(boundaryFields<scalar>& bfield, Field<scalar>& internalField) {

    };

    int size() const
    {
        return size_;
    }

protected:

    const unstructuredMesh& mesh_;
    int patchID_;
    int start_;
    int end_;
    int size_;
};

};
