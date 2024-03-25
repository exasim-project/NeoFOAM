#include "NeoFOAM/cellCentredFiniteVolume/bcFields/scalar/fvccScalarEmptyBoundaryField.hpp"

namespace NeoFOAM
{
fvccScalarEmptyBoundaryField::fvccScalarEmptyBoundaryField(const unstructuredMesh& mesh, int patchID)
    : fvccBoundaryField<scalar>(mesh, patchID)
{
}

void fvccScalarEmptyBoundaryField::correctBoundaryConditions(boundaryFields<scalar>& bfield, const Field<scalar>& internalField)
{
    // do nothing
}


}