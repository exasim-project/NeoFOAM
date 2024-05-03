// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/surface/scalar/fvccSurfaceScalarCalculatedBoundaryField.hpp"

namespace NeoFOAM
{

fvccSurfaceScalarCalculatedBoundaryField::fvccSurfaceScalarCalculatedBoundaryField(const unstructuredMesh& mesh, int patchID)
    : fvccSurfaceBoundaryField<scalar>(mesh, patchID)
{
}

void fvccSurfaceScalarCalculatedBoundaryField::correctBoundaryConditions(boundaryFields<scalar>& bfield, Field<scalar>& internalField)
{
}


}
