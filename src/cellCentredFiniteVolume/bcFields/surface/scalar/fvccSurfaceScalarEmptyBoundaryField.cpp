// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/surface/scalar/fvccSurfaceScalarEmptyBoundaryField.hpp"

namespace NeoFOAM
{
fvccSurfaceScalarEmptyBoundaryField::fvccSurfaceScalarEmptyBoundaryField(const unstructuredMesh& mesh, int patchID)
    : fvccSurfaceBoundaryField<scalar>(mesh, patchID)
{
}

void fvccSurfaceScalarEmptyBoundaryField::correctBoundaryConditions(boundaryFields<scalar>& bfield, Field<scalar>& internalField)
{
    // do nothing
}


}