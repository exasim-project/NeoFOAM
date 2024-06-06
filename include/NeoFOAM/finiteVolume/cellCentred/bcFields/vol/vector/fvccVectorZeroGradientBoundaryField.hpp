// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once
#include "Kokkos_Core.hpp"

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "NeoFOAM/mesh/unstructured/UnstructuredMesh.hpp"

namespace NeoFOAM
{
struct ZeroGradientVectorBCKernel
{
    const UnstructuredMesh& mesh_;
    int patchID_;
    int start_;
    int end_;

    ZeroGradientVectorBCKernel(const UnstructuredMesh& mesh, int patchID, int start, int end)
        : mesh_(mesh), patchID_(patchID), start_(start), end_(end)
    {
        // TODO add asserts end > start
    }

    void operator()(
        const GPUExecutor& exec, BoundaryFields<Vector>& bField, const Field<Vector>& internalField
    );

    void operator()(
        const OMPExecutor& exec, BoundaryFields<Vector>& bField, const Field<Vector>& internalField
    );

    void operator()(
        const CPUExecutor& exec, BoundaryFields<Vector>& bField, const Field<Vector>& internalField
    );
};

class fvccVectorZeroGradientBoundaryField : public fvccBoundaryField<Vector>
{
public:

    fvccVectorZeroGradientBoundaryField(const UnstructuredMesh& mesh, int patchID);

    virtual void
    correctBoundaryConditions(BoundaryFields<Vector>& bfield, const Field<Vector>& internalField);

private:
};
};
