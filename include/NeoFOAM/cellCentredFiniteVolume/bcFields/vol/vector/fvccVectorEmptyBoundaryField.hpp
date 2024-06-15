// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "Kokkos_Core.hpp"

#include "NeoFOAM/core.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"
#include "NeoFOAM/mesh/unstructured.hpp"

namespace NeoFOAM
{

class fvccVectorEmptyBoundaryField : public fvccBoundaryField<Vector>
{
public:

    fvccVectorEmptyBoundaryField(const UnstructuredMesh& mesh, int patchID);

    virtual void
    correctBoundaryConditions(BoundaryFields<Vector>& bfield, const Field<Vector>& internalField);

private:
};
};
