// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/primitives/scalar.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "Kokkos_Core.hpp"
#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{
struct ZeroGradientVectorBCKernel
{
    const unstructuredMesh& mesh_;
    int patchID_;
    int start_;
    int end_;

    void operator()(const GPUExecutor& exec, boundaryFields<Vector>& bField, const Field<Vector>& internalField);

    void operator()(const OMPExecutor& exec, boundaryFields<Vector>& bField, const Field<Vector>& internalField);

    void operator()(const CPUExecutor& exec, boundaryFields<Vector>& bField, const Field<Vector>& internalField);
};

class fvccVectorZeroGradientBoundaryField : public fvccBoundaryField<Vector>
{
public:

    fvccVectorZeroGradientBoundaryField(const unstructuredMesh& mesh, int patchID);

    virtual void correctBoundaryConditions(boundaryFields<Vector>& bfield, const Field<Vector>& internalField);

private:
};
};