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
struct fixedVectorValueBCKernel
{
    const UnstructuredMesh& mesh_;
    int patchID_;
    int start_;
    int end_;
    Vector uniformValue_;

    fixedVectorValueBCKernel(
        const UnstructuredMesh& mesh, int patchID, int start, int end, Vector uniformValue
    )
        : mesh_(mesh), patchID_(patchID), start_(start), end_(end), uniformValue_(uniformValue)
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

class fvccVectorFixedValueBoundaryField : public fvccBoundaryField<Vector>
{
public:

    fvccVectorFixedValueBoundaryField(
        const UnstructuredMesh& mesh, int patchID, Vector uniformValue
    );

    virtual void
    correctBoundaryConditions(BoundaryFields<Vector>& bfield, const Field<Vector>& internalField);

private:

    Vector uniformValue_;
};
};
