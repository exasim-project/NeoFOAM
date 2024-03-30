#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarZeroGradientBoundaryField.hpp"

namespace NeoFOAM
{
fvccScalarZeroGradientBoundaryField::fvccScalarZeroGradientBoundaryField(const unstructuredMesh& mesh, int patchID)
    : fvccBoundaryField<scalar>(mesh, patchID)
{
}

void fvccScalarZeroGradientBoundaryField::correctBoundaryConditions(boundaryFields<scalar>& bfield, const Field<scalar>& internalField)
{

}

}