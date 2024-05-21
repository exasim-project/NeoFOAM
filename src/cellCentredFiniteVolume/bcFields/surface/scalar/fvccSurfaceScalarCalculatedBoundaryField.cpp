// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/surface/scalar/fvccSurfaceScalarCalculatedBoundaryField.hpp"

namespace NeoFOAM
{

fvccSurfaceScalarCalculatedBoundaryField::fvccSurfaceScalarCalculatedBoundaryField(
    const UnstructuredMesh& mesh, int patchID
)
    : fvccSurfaceBoundaryField<scalar>(mesh, patchID)
{}

void fvccSurfaceScalarCalculatedBoundaryField::correctBoundaryConditions(
    BoundaryFields<scalar>& bfield, Field<scalar>& internalField
)
{}


}
