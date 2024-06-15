// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once
#include "Kokkos_Core.hpp"

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM
{

class fvccVectorCalculatedBoundaryField : public fvccBoundaryField<Vector>
{
public:

    fvccVectorCalculatedBoundaryField(const UnstructuredMesh& mesh, int patchID);

    virtual void
    correctBoundaryConditions(BoundaryFields<Vector>& bfield, const Field<Vector>& internalField);

private:
};
};
