#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvsPatch/scalar/fvccSurfaceScalarCalculatedBoundaryField.hpp"

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