// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once
#include "Kokkos_Core.hpp"

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM
{


class fvccSurfaceScalarCalculatedBoundaryField : public fvccSurfaceBoundaryField<scalar>
{
public:

    fvccSurfaceScalarCalculatedBoundaryField(const UnstructuredMesh& mesh, int patchID);

    virtual void
    correctBoundaryConditions(BoundaryFields<scalar>& bfield, Field<scalar>& internalField);

private:
};
};
