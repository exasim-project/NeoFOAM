// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/primitives/scalar.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccSurfaceBoundaryField.hpp"
#include "Kokkos_Core.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{

class fvccSurfaceScalarEmptyBoundaryField : public fvccSurfaceBoundaryField<scalar>
{
public:

    fvccSurfaceScalarEmptyBoundaryField(const unstructuredMesh& mesh, int patchID);

    virtual void correctBoundaryConditions(boundaryFields<scalar>& bfield, Field<scalar>& internalField);

private:
};
};
