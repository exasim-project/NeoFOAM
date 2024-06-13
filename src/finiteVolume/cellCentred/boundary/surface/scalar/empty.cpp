// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/surface/scalar/fvccSurfaceScalarEmptyBoundaryField.hpp"

namespace NeoFOAM
{
fvccSurfaceScalarEmptyBoundaryField::fvccSurfaceScalarEmptyBoundaryField(
    const UnstructuredMesh& mesh, int patchID
)
    : fvccSurfaceBoundaryField<scalar>(mesh, patchID)
{}

void fvccSurfaceScalarEmptyBoundaryField::correctBoundaryConditions(
    BoundaryFields<scalar>& bfield, Field<scalar>& internalField
)
{
    // do nothing
}


}
