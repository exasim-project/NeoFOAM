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
struct fixedValueBCKernel
{
    const unstructuredMesh& mesh_;
    int patchID_;
    int start_;
    int end_;
    scalar uniformValue_;

    void operator()(const GPUExecutor& exec, boundaryFields<scalar>& bField, const Field<scalar>& internalField);

    void operator()(const OMPExecutor& exec, boundaryFields<scalar>& bField, const Field<scalar>& internalField);

    void operator()(const CPUExecutor& exec, boundaryFields<scalar>& bField, const Field<scalar>& internalField);
};

class fvccScalarFixedValueBoundaryField : public fvccBoundaryField<scalar>
{
public:

    fvccScalarFixedValueBoundaryField(const unstructuredMesh& mesh, int patchID, scalar uniformValue);

    virtual void correctBoundaryConditions(boundaryFields<scalar>& bfield, const Field<scalar>& internalField);

private:

    scalar uniformValue_;
};
};