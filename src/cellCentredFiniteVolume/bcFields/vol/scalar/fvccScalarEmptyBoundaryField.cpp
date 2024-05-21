// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarEmptyBoundaryField.hpp"

namespace NeoFOAM
{
fvccScalarEmptyBoundaryField::fvccScalarEmptyBoundaryField(
    const UnstructuredMesh& mesh, int patchID
)
    : fvccBoundaryField<scalar>(mesh, patchID)
{}

void fvccScalarEmptyBoundaryField::correctBoundaryConditions(
    BoundaryFields<scalar>& bfield, const Field<scalar>& internalField
)
{
    // do nothing
}


}
