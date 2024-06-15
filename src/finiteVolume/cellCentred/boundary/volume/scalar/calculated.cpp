// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred.hpp"

namespace NeoFOAM
{
fvccScalarCalculatedBoundaryField::fvccScalarCalculatedBoundaryField(
    const UnstructuredMesh& mesh, int patchID
)
    : fvccBoundaryField<scalar>(mesh, patchID)
{}

void fvccScalarCalculatedBoundaryField::correctBoundaryConditions(
    BoundaryFields<scalar>& bfield, const Field<scalar>& internalField
)
{}

}
