// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/primitives/scalar.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvsPatch/fvccSurfaceBoundaryField.hpp"
#include "Kokkos_Core.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{

class fvccScalarEmptyBoundaryField : public fvccSurfaceBoundaryField<scalar>
{
public:

    fvccScalarEmptyBoundaryField(const unstructuredMesh& mesh, int patchID);

    virtual void correctBoundaryConditions(boundaryFields<scalar>& bfield, const Field<scalar>& internalField);

private:
};
};