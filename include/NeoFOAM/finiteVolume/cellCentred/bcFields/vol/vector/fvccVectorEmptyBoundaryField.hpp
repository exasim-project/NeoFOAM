// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "Kokkos_Core.hpp"

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "NeoFOAM/mesh/unstructured/UnstructuredMesh.hpp"

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
