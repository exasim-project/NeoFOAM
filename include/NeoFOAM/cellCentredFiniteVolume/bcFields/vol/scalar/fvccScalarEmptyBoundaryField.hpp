// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once
#include "Kokkos_Core.hpp"

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"

namespace NeoFOAM
{

class fvccScalarEmptyBoundaryField : public fvccBoundaryField<scalar>
{
public:

    fvccScalarEmptyBoundaryField(const UnstructuredMesh& mesh, int patchID);

    virtual void
    correctBoundaryConditions(BoundaryFields<scalar>& bfield, const Field<scalar>& internalField);

private:
};
};
