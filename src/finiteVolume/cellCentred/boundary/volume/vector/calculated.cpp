// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/vector/fvccVectorCalculatedBoundaryField.hpp"

namespace NeoFOAM
{
fvccVectorCalculatedBoundaryField::fvccVectorCalculatedBoundaryField(
    const UnstructuredMesh& mesh, int patchID
)
    : fvccBoundaryField<Vector>(mesh, patchID)
{}

void fvccVectorCalculatedBoundaryField::correctBoundaryConditions(
    BoundaryFields<Vector>& bfield, const Field<Vector>& internalField
)
{}


}
