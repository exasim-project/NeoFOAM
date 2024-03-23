// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/primitives/scalar.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "Kokkos_Core.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{
struct ZeroGradientBCKernel
{
    const unstructuredMesh& mesh_;
    int patchID_;
    int start_;
    int end_;

    void operator()(const GPUExecutor& exec, boundaryFields<scalar>& bField, const Field<scalar>& internalField);

    void operator()(const OMPExecutor& exec, boundaryFields<scalar>& bField, const Field<scalar>& internalField);

    void operator()(const CPUExecutor& exec, boundaryFields<scalar>& bField, const Field<scalar>& internalField);
};

class fvccScalarZeroGradientBoundaryField : public fvccBoundaryField<scalar>
{
public:

    fvccScalarZeroGradientBoundaryField(const unstructuredMesh& mesh, int patchID);

    virtual void correctBoundaryConditions(boundaryFields<scalar>& bfield, const Field<scalar>& internalField);

private:
};
};