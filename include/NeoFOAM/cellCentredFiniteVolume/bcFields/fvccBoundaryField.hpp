// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"
#include "NeoFOAM/fields/boundaryFields.hpp"

namespace NeoFOAM
{

template<typename T>
class fvccBoundaryField
{

public:

    fvccBoundaryField(int start, int end)
        : start_(start),
          end_(end),
          size_(end - start)

              {

              };

    virtual void correctBoundaryConditions(boundaryFields<T>& field) {

    };

protected:

    int size() const {
        return size_;
    }

    int start_;
    int end_;
    int size_;
};
};