#include "NeoFOAM/cellCentredFiniteVolume/bcFields/scalar/fvccScalarZeroGradientBoundaryField.hpp"

namespace NeoFOAM
{
fvccScalarZeroGradientBoundaryField::fvccScalarZeroGradientBoundaryField(const unstructuredMesh& mesh, int patchID)
    : fvccBoundaryField<scalar>(mesh, patchID)
{
}

void fvccScalarZeroGradientBoundaryField::correctBoundaryConditions(boundaryFields<scalar>& bfield, const Field<scalar>& internalField)
{

}


void ZeroGradientBCKernel::operator()(const GPUExecutor& exec, boundaryFields<scalar>& bField, const Field<scalar>& internalField)
{

}

void ZeroGradientBCKernel::operator()(const OMPExecutor& exec, boundaryFields<scalar>& bField, const Field<scalar>& internalField)
{

}

void ZeroGradientBCKernel::operator()(const CPUExecutor& exec, boundaryFields<scalar>& bField, const Field<scalar>& internalField)
{

}

}