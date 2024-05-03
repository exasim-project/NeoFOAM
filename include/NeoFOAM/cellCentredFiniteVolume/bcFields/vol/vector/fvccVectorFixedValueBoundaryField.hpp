// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/primitives/scalar.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"
#include "Kokkos_Core.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{
struct fixedVectorValueBCKernel
{
    const unstructuredMesh& mesh_;
    int patchID_;
    int start_;
    int end_;
    Vector uniformValue_;

    void operator()(const GPUExecutor& exec, boundaryFields<Vector>& bField, const Field<Vector>& internalField);

    void operator()(const OMPExecutor& exec, boundaryFields<Vector>& bField, const Field<Vector>& internalField);

    void operator()(const CPUExecutor& exec, boundaryFields<Vector>& bField, const Field<Vector>& internalField);
};

class fvccVectorFixedValueBoundaryField : public fvccBoundaryField<Vector>
{
public:

    fvccVectorFixedValueBoundaryField(const unstructuredMesh& mesh, int patchID, Vector uniformValue);

    virtual void correctBoundaryConditions(boundaryFields<Vector>& bfield, const Field<Vector>& internalField);

private:

    Vector uniformValue_;
};
};
