#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarCalculatedBoundaryField.hpp"

namespace NeoFOAM
{
fvccScalarCalculatedBoundaryField::fvccScalarCalculatedBoundaryField(const unstructuredMesh& mesh, int patchID)
    : fvccBoundaryField<scalar>(mesh, patchID)
{
}

void fvccScalarCalculatedBoundaryField::correctBoundaryConditions(boundaryFields<scalar>& bfield, const Field<scalar>& internalField)
{

}

}